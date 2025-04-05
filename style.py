import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pickle


class FaceStyleTransfer:
    def __init__(self, device=None):
        """
        Initialize the face style transfer model
        
        Args:
            device (torch.device): Device to use for computation
        """
        # Set device (use GPU if available)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Image loading and preprocessing
        self.imsize = 512 if torch.cuda.is_available() else 384
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Define the layers of interest in VGG19
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Set up the feature extraction model
        self.model = self.get_model()
        
        # Load face detection model from OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize style data
        self.style_img = None
        self.style_features = None
        
    def load_image(self, path, resize=True):
        """Load an image and convert to PIL"""
        image = Image.open(path).convert('RGB')
        
        if resize:
            # Resize while preserving aspect ratio
            width, height = image.size
            max_size = max(width, height)
            scale = self.imsize / max_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Return PIL image for further processing
        return image
    
    def preprocess_image(self, image):
        """Convert PIL image to normalized tensor"""
        if isinstance(image, str):
            # If a path is provided instead of an image
            image = self.load_image(image)
            
        # Convert PIL image to tensor
        image_tensor = self.loader(image).unsqueeze(0)
        return image_tensor.to(self.device, torch.float)
    
    def detect_face(self, image):
        """
        Detect face in the image using OpenCV cascade classifier
        
        Args:
            image: PIL image or path to image
            
        Returns:
            tuple: (face_image, face_location)
        """
        if isinstance(image, str):
            image = self.load_image(image, resize=False)
        
        # Convert PIL image to OpenCV format (BGR)
        img_np = np.array(image)
        img_cv = img_np[:, :, ::-1].copy()  # RGB to BGR
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            print("No face detected!")
            return None, None
        
        # Use the largest face if multiple are detected
        if len(faces) > 1:
            areas = [w * h for (x, y, w, h) in faces]
            max_idx = np.argmax(areas)
            x, y, w, h = faces[max_idx]
        else:
            x, y, w, h = faces[0]
        
        # Add some margin (20% of face size)
        margin_w, margin_h = int(w * 0.2), int(h * 0.2)
        
        # Ensure margins don't go outside the image bounds
        left = max(0, x - margin_w)
        top = max(0, y - margin_h)
        right = min(img_np.shape[1], x + w + margin_w)
        bottom = min(img_np.shape[0], y + h + margin_h)
        
        # Extract the face region from the PIL image
        face_img = image.crop((left, top, right, bottom))
        
        # Return the face image and location
        return face_img, (top, right, bottom, left)
    
    def create_face_mask(self, image, face_loc):
        """
        Create a binary mask for the face region using the face location
        
        Args:
            image: PIL image
            face_loc: tuple of (top, right, bottom, left)
            
        Returns:
            PIL.Image: Binary mask highlighting the face
        """
        if face_loc is None:
            return None
            
        # Get image dimensions
        width, height = image.size
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Extract face location
        top, right, bottom, left = face_loc
        
        # Create elliptical mask for the face region
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        axis_x = (right - left) // 2
        axis_y = (bottom - top) // 2
        
        # Create coordinate grid
        y, x = np.ogrid[:height, :width]
        
        # Create elliptical mask
        mask_ellipse = ((x - center_x)**2 / (axis_x**2) + (y - center_y)**2 / (axis_y**2)) <= 1
        mask[mask_ellipse] = 255
        
        # Blur the mask to soften edges
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Convert to PIL image
        mask_img = Image.fromarray(mask)
        
        return mask_img
    
    def get_model(self):
        """Create VGG19 model for feature extraction"""
        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Create a sequential model with non-inplace ReLU
        model = nn.Sequential()
        
        # Modify the VGG model to use non-inplace ReLU
        for i, layer in enumerate(vgg.children()):
            if isinstance(layer, nn.ReLU):
                # Replace in-place version with out-of-place
                model.add_module(str(i), nn.ReLU(inplace=False))
            else:
                model.add_module(str(i), layer)
        
        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def get_features(self, image, model):
        """Extract features from intermediate layers of the model"""
        features = {}
        layer_names = []
        x = image
        
        # Layer mapping for VGG19
        layer_mapping = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
        
        # Add all the layers to layer_names
        for name, module in model._modules.items():
            x = module(x)
            # Check if the current layer is one of the layers of interest
            if name in layer_mapping:
                layer_name = layer_mapping[name]
                features[layer_name] = x
                layer_names.append(layer_name)
        
        return features, layer_names
    
    def gram_matrix(self, input_tensor):
        """Calculate Gram matrix for a given tensor"""
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size, channels, height * width)
        
        # Compute the gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by the number of elements in each feature map
        return gram.div(channels * height * width)
    
    def compute_content_loss(self, target_features, content_features):
        """Compute content loss between target and content features"""
        content_loss = 0
        for layer in self.content_layers:
            if layer in target_features and layer in content_features:
                target_feature = target_features[layer]
                content_feature = content_features[layer]
                content_loss += F.mse_loss(target_feature, content_feature)
        
        return content_loss
    
    def compute_style_loss(self, target_features, style_features, style_weights=None):
        """Compute style loss between target and style features"""
        if style_weights is None:
            # Equal weights for all style layers
            style_weights = {layer: 1.0 / len(self.style_layers) for layer in self.style_layers}
        
        style_loss = 0
        for layer in self.style_layers:
            if layer not in target_features or layer not in style_features:
                continue
                
            target_feature = target_features[layer]
            style_feature = style_features[layer]
            
            target_gram = self.gram_matrix(target_feature)
            style_gram = self.gram_matrix(style_feature)
            
            layer_style_loss = F.mse_loss(target_gram, style_gram) * style_weights[layer]
            style_loss += layer_style_loss
        
        return style_loss
    
    def load_style_image(self, style_img_path):
        """
        Load a style image and process it for later use
        
        Args:
            style_img_path: Path to the style image
            
        Returns:
            The processed style image
        """
        print(f"Loading style image from {style_img_path}")
        
        # Load the style image
        style_full = self.load_image(style_img_path, resize=False)
        
        # Detect face in the style image
        style_face, _ = self.detect_face(style_full)
        
        # If no face detected, use the whole image
        if style_face is None:
            print("No face detected in style image, using whole image")
            style_face = style_full
        
        # Resize style face to working size
        style_face_resized = style_face.resize((self.imsize, self.imsize), Image.LANCZOS)
        
        # Convert to tensor
        style_tensor = self.preprocess_image(style_face_resized)
        
        # Extract features
        style_features, _ = self.get_features(style_tensor, self.model)
        
        # Store the style data
        self.style_img = style_tensor
        self.style_features = style_features
        
        return style_face_resized
    
    def save_style_model(self, path):
        """
        Save the extracted style features for later use
        
        Args:
            path: Path where to save the model
        """
        if self.style_features is None:
            raise ValueError("No style features to save. Load a style image first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Extract style gram matrices for more efficient storage
        style_grams = {}
        for layer in self.style_layers:
            if layer in self.style_features:
                style_grams[layer] = self.gram_matrix(self.style_features[layer]).cpu().detach()
        
        # Save style data
        state = {
            'style_grams': style_grams,
            'content_layers': self.content_layers,
            'style_layers': self.style_layers,
            'imsize': self.imsize
        }
        
        torch.save(state, path)
        print(f"Style model saved to {path}")
    
    def load_style_model(self, path):
        """
        Load a previously saved style model
        
        Args:
            path: Path to the saved style model
        """
        print(f"Loading style model from {path}")
        
        # Load the saved state
        state = torch.load(path, map_location=self.device)
        
        # Set parameters from saved state
        self.content_layers = state['content_layers']
        self.style_layers = state['style_layers']
        self.imsize = state['imsize']
        
        # Initialize style features
        self.style_features = {}
        
        # Convert gram matrices to feature format
        for layer, gram in state['style_grams'].items():
            self.style_features[layer] = gram.to(self.device)
        
        # Create a dummy style image
        self.style_img = torch.ones((1, 3, self.imsize, self.imsize), device=self.device)
        
        print(f"Style model loaded successfully")
    
    def transfer_style_to_face(self, content_img_path, output_path=None, 
                             num_steps=300, content_weight=1e5, style_weight=1e10,
                             face_regions_only=True):
        """
        Apply saved style to content image
        
        Args:
            content_img_path: Path to content image (target face)
            output_path: Path to save the output image
            num_steps: Number of optimization steps
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            face_regions_only: Whether to apply style to face regions only
        
        Returns:
            PIL.Image: The stylized image
        """
        if self.style_features is None:
            raise ValueError("No style loaded. Call load_style_image() or load_style_model() first.")
            
        print("Starting face style transfer...")
        
        # Load content image
        content_full = self.load_image(content_img_path, resize=False)
        
        # Detect face in the content image
        content_face, content_loc = self.detect_face(content_full)
        
        if content_face is None:
            print("No face detected in content image, using whole image")
            content_face = content_full
            face_regions_only = False
        
        # Create face mask if needed
        if face_regions_only and content_loc:
            content_mask = self.create_face_mask(content_face, content_loc)
            
            # Resize content face to working size
            content_face_resized = content_face.resize(
                (self.imsize, self.imsize), Image.LANCZOS)
            content_mask_resized = content_mask.resize(
                (self.imsize, self.imsize), Image.LANCZOS)
        else:
            # Use the entire image
            content_face_resized = content_face.resize(
                (self.imsize, self.imsize), Image.LANCZOS)
            content_mask_resized = None
        
        # Convert to tensor
        content_tensor = self.preprocess_image(content_face_resized)
        
        # Create a copy of the content image to optimize
        input_img = content_tensor.clone()
        
        # Extract content features
        content_features, _ = self.get_features(content_tensor, self.model)
        
        # Set requires_grad to True for optimization
        input_img.requires_grad_(True)
        
        # Setup optimizer (LBFGS is better for style transfer but can be memory-intensive)
        optimizer = optim.LBFGS([input_img], max_iter=20)
        
        # Track progress
        step = [0]  # Using list to modify inside closure
        all_images = []
        
        print("Optimizing...")
        while step[0] < num_steps:
            def closure():
                optimizer.zero_grad()
                
                # Forward pass through the model
                features, _ = self.get_features(input_img, self.model)
                
                # Compute losses
                content_score = self.compute_content_loss(features, content_features)
                
                # For style loss, check if we have gram matrices already or feature maps
                if isinstance(next(iter(self.style_features.values())), torch.Tensor) and next(iter(self.style_features.values())).dim() == 3:
                    # We have gram matrices already (from loaded model)
                    style_score = 0
                    for layer in self.style_layers:
                        if layer not in features or layer not in self.style_features:
                            continue
                        
                        target_gram = self.gram_matrix(features[layer])
                        style_gram = self.style_features[layer]  # Already a gram matrix
                        
                        style_score += F.mse_loss(target_gram, style_gram) * (1.0 / len(self.style_layers))
                else:
                    # We have feature maps, compute style loss normally
                    style_score = self.compute_style_loss(features, self.style_features)
                
                # Weight the losses
                content_loss = content_weight * content_score
                style_loss = style_weight * style_score
                
                # Total loss
                loss = content_loss + style_loss
                
                # Backward pass
                loss.backward()
                
                # Log progress
                if step[0] % 50 == 0 or step[0] == num_steps - 1:
                    print(f"Step {step[0]}/{num_steps}:")
                    print(f"Style Loss: {style_loss.item():.4f}, Content Loss: {content_loss.item():.4f}")
                    
                    # Save intermediate result
                    result_img = self.tensor_to_image(input_img.clone())
                    all_images.append(result_img)
                
                step[0] += 1
                return loss
            
            optimizer.step(closure)
        
        # Get final result
        result_tensor = input_img.clone()
        result_img = self.tensor_to_image(result_tensor)
        
        # Apply the result back to the original image if using face regions
        if face_regions_only and content_loc and content_mask_resized:
            # Ensure the mask is in the right format
            mask_np = np.array(content_mask_resized) / 255.0
            
            # Convert result to numpy array
            result_np = np.array(result_img)
            
            # Convert original content face to numpy
            content_face_np = np.array(content_face_resized)
            
            # Create a blended result using the mask
            mask_np = mask_np[:, :, np.newaxis]  # Add channel dimension
            blended = result_np * mask_np + content_face_np * (1 - mask_np)
            blended_img = Image.fromarray(blended.astype(np.uint8))
            
            # Resize back to original content face size
            blended_img = blended_img.resize(content_face.size, Image.LANCZOS)
            
            # Paste back to the original image
            top, right, bottom, left = content_loc
            content_full_copy = content_full.copy()
            content_full_copy.paste(blended_img, (left, top))
            final_result = content_full_copy
        else:
            # Use the entire stylized image
            # Resize to original content image size
            final_result = result_img.resize(content_full.size, Image.LANCZOS)
        
        # Save the result if output path is provided
        if output_path:
            final_result.save(output_path)
            print(f"Result saved to {output_path}")
        
        return final_result, all_images
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL image"""
        # Make a copy of the tensor
        tensor = tensor.clone().detach()
        
        # Remove the batch dimension
        tensor = tensor.squeeze(0)
        
        # Unnormalize
        tensor = tensor.cpu().clone()
        
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        tensor = tensor * std[:, None, None] + mean[:, None, None]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        image = transforms.ToPILImage()(tensor)
        return image
    
    def plot_results(self, content_img, style_img, result_img, all_images=None):
        """
        Visualize the style transfer results
        
        Args:
            content_img: PIL image or path to content image
            style_img: PIL image or path to style image
            result_img: PIL image of the result
            all_images: List of intermediate result images
        """
        if isinstance(content_img, str):
            content_img = self.load_image(content_img)
        
        if isinstance(style_img, str):
            style_img = self.load_image(style_img)
        
        # Create a figure
        plt.figure(figsize=(15, 10))
        
        # Plot content image
        plt.subplot(2, 3, 1)
        plt.imshow(content_img)
        plt.title('Content Image')
        plt.axis('off')
        
        # Plot style image
        plt.subplot(2, 3, 2)
        plt.imshow(style_img)
        plt.title('Style Image')
        plt.axis('off')
        
        # Plot result
        plt.subplot(2, 3, 3)
        plt.imshow(result_img)
        plt.title('Result')
        plt.axis('off')
        
        # Plot intermediate results
        if all_images and len(all_images) > 0:
            # Select a few intermediate results
            n_images = min(5, len(all_images))
            indices = np.linspace(0, len(all_images)-1, n_images).astype(int)
            
            plt.subplot(2, 3, 4)
            grid_img = Image.new('RGB', (self.imsize * n_images, self.imsize))
            
            for i, idx in enumerate(indices):
                img = all_images[idx].resize((self.imsize, self.imsize), Image.LANCZOS)
                grid_img.paste(img, (i * self.imsize, 0))
            
            plt.imshow(grid_img)
            plt.title('Optimization Progress')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def train_save_style_model(style_path, model_save_path):
    """
    Train a style model and save it
    
    Args:
        style_path: Path to the style image
        model_save_path: Path to save the model
    """
    # Initialize face style transfer
    model = FaceStyleTransfer()
    
    # Load and process style image
    model.load_style_image(style_path)
    
    # Save the model
    model.save_style_model(model_save_path)
    
    print(f"Style model trained and saved to {model_save_path}")
    
    return model


def apply_saved_style_to_face(content_path, model_path, output_path=None, 
                            num_steps=300, content_weight=1e5, style_weight=1e10,
                            face_regions_only=True, show_result=True):
    """
    Apply a previously saved style model to a content image
    
    Args:
        content_path: Path to the content image (target face)
        model_path: Path to the saved style model
        output_path: Path to save the output image
        num_steps: Number of optimization steps
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        face_regions_only: Whether to apply style to face regions only
        show_result: Whether to display the result
        
    Returns:
        PIL.Image: The stylized image
    """
    # Initialize face style transfer
    model = FaceStyleTransfer()
    
    # Load the style model
    model.load_style_model(model_path)
    
    # Apply style transfer
    result_img, all_images = model.transfer_style_to_face(
        content_path, 
        output_path=output_path,
        num_steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight,
        face_regions_only=face_regions_only
    )
    
    # Display the result
    if show_result and model.style_img is not None:
        # For loaded models, we don't have the original style image
        # Create a placeholder or load it separately if available
        style_img = Image.new('RGB', (100, 100), color=(200, 200, 200))
        model.plot_results(content_path, style_img, result_img, all_images)
    
    return result_img


if __name__ == "__main__":
    # Example 1: Train and save a style model
    style_path = "korean.png"
    model_save_path = "models/my_style_model.pt"
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train and save the style model
    model = train_save_style_model(style_path, model_save_path)
    
    # Example 2: Apply the saved style model to a content image
    content_path = "white.png"
    output_path = "stylized_face.jpg"
    
    # Apply the saved style model
    result = apply_saved_style_to_face(
        content_path,
        model_save_path,
        output_path,
        num_steps=1000,
        content_weight=1e5,
        style_weight=1e10,
        face_regions_only=True,
        show_result=True
    )
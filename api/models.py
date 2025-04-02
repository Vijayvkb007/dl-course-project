from django.db import models
from datetime import datetime

class Image(models.Model):
    """Image model to store the RGB and IR images, and its predictions.

        rgb_image: ImageField to store the RGB image.
        ir_image: ImageField to store the IR image.
        prediction: JSONField to store the prediction.
        created_at: DateTimeField to store the creation time of the image.
    """
    rgb_image = models.ImageField(upload_to='rgb_images/')
    ir_image = models.ImageField(upload_to='ir_images/')
    prediction = models.JSONField(null=True, blank=True)    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        # Delete old images if they exist
        try:
            existing = Image.objects.latest('created_at')
            if existing.rgb_image:
                existing.rgb_image.delete(save=False)
            if existing.ir_image:
                existing.ir_image.delete(save=False)
            if existing.prediction:
                existing.prediction.clear()
        except Image.DoesNotExist:
            pass
       
        timestamp = datetime.now().strftime('%d-%m-%y__%H-%M-%S')
        # Set fixed names for the images
        self.rgb_image.name = f'rgb_image_{timestamp}.jpg'
        self.ir_image.name = f'ir_image_{timestamp}.jpg'

        # save these images
        super().save(*args, **kwargs)
        
        # get the prediction
        from .utils import prediction
        try:
            self.prediction = prediction()
            super().save(update_fields=['prediction'])
        except Exception as e:
            self.prediction = {"error": str(e)}
            print(f"Error: {e}")
            super().save(update_fields=['prediction'])

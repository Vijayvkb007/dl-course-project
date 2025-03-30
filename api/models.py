from django.db import models

class Image(models.Model):
    rgb_image = models.ImageField(upload_to='rgb_images/')
    ir_image = models.ImageField(upload_to='ir_images/')    
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
        except Image.DoesNotExist:
            pass
        
        # Set fixed names for the images
        self.rgb_image.name = 'rgb_image.jpg'
        self.ir_image.name = 'ir_image.jpg'
        
        super().save(*args, **kwargs)

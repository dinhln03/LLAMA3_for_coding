from django.db import models
from django.conf import settings


# Create your models here.


class Message(models.Model):
    id = models.AutoField(primary_key=True)
    sender_id = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="message_sender", on_delete=models.DO_NOTHING, null=True)
    receiver_id = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="message_receiver", on_delete=models.DO_NOTHING, null=True)
    content = models.TextField(null=True)
    read_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']


class Follower(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="following", on_delete=models.CASCADE, null=True)
    follower_id = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="follower")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.user_id.username

    class Meta:
        ordering = ['user_id']


class PostCategory(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Post Categories"
        ordering = ['name']

    def __str__(self):
        return self.name


class Post(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET(None))
    title = models.CharField(max_length=50, null=True)
    post_category_id = models.ForeignKey(PostCategory, null=True, on_delete=models.SET_NULL, blank=True)
    content = models.TextField(null=True)
    likes = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True, related_name="likes")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return str(self.title)


class PostComment(models.Model):
    id = models.AutoField(primary_key=True)
    post_id = models.ForeignKey(Post, on_delete=models.CASCADE)
    commenter_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    content = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Post Comments"
        ordering = ['-created_at']


class PostLike(models.Model):
    id = models.AutoField(primary_key=True)
    liker_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    post_id = models.ForeignKey(Post, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Post Likes"


from django.db import models
from django.contrib.auth import get_user_model


User = get_user_model() 

class Group(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()

class Post(models.Model):
    text = models.TextField()
    pub_date = models.DateTimeField("date published", auto_now_add=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="post_author")
    group = models.ForeignKey(Group, on_delete=models.CASCADE, blank=True, null=True)
    image = models.ImageField(upload_to='posts/', blank=True, null=True)


class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    text = models.TextField()
    created = models.DateTimeField('Дата и время публикации', auto_now_add=True, db_index=True)
    
    def __str__(self):
       return self.text


class Follow(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='follower') #тот который подписывается
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='following') #тот на которого подписываются

    def __str__(self):
       return self.text

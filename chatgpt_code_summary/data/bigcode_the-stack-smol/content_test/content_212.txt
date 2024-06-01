from django.forms import ModelForm
from .models import Post, Comment
from loginsignup.utils import getBeaverInstance


class PostForm(ModelForm):
    class Meta:
        model = Post
        exclude = ["likes", "posted_on", "post_creator"]

    def checkPost(self, request):
        if self.is_valid():
            post = self.save(commit=False)
            beaver = getBeaverInstance(request)
            post.post_creator = beaver
            post.save()
            return True
        return False


class CommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ["comment"]

    def checkComment(self, request, post):
        if self.is_valid():
            comment = self.save(commit=False)
            comment.comment_creator = getBeaverInstance(request)
            comment.post = post
            comment.save()
            return True
        return False

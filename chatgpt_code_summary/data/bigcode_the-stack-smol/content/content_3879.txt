"""Views of problem2 app."""

from django.shortcuts import render

from .forms import FiboForm


def display(request):
    """Function view to display form in the standard manner."""
    if request.method == 'POST':
        form = FiboForm(request.POST)
        if form.is_valid():
            fibo = form.save(commit=False)
            evensum = fibo.evenFiboSum()
            fibo.save()
            return render(request, 'problem2/solution2.html',
                          {'evensum': evensum, 'form': form})
    else:
        form = FiboForm()
    return render(request, 'problem2/solution2.html', {'form': form})

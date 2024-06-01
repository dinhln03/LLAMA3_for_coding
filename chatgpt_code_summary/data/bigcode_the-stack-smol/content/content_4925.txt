"""
list of movies that feed into fresh_tomatoes.py file
"""
import fresh_tomatoes
from get_movie_list import get_movie_list

def main():
    """
        Main entry point for the script.
    """
    # Read in the movies from the json file
    movie_list = get_movie_list("src/data/movies.json")

    # Generate the html file and display in a browser window
    fresh_tomatoes.open_movies_page(movie_list)

main()

"""
A program to recommend which movie a user should watch next, based on the similarity of the movie 
decription they have watched.This module uses Spacy Model to perform text similarity comparison
and then most similar movie is return

functions:

    * get_movie_df -> create and returns a dataframe of movies 
    * recsys-> returns similarity scores performed
    * main -> the main function of the script
"""

import spacy
import pandas as pd

user_data = {'Planet Hulk': "Will he save thier world or destory it?\
             When the Hulk becomes too dangerous for the Earth, \
             the illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace.\
             Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery where he is sold into \
             slaveryand trained as a gladiator."}

def get_movie_df():
    """
    creates a dataframe using the movies text file and returns it
    Parameters
    ----------
    None

    Returns
    -------
    dataframe
        a dataframe of the movies consisting of the moives title and thier descriptions
    """
    movies_df = pd.read_csv('movies.txt', sep= ':', names= ['Movie Title', 'Description'])
    return movies_df

def recsys(model, user_movie_desc, movies_desc):
    """
    runs a simalarity comparision and returns the similarity scores
    Parameters
    ----------
    model : object
        a spacy model object used to perform similarity comparisons 
    user_movie_desc: str
        a descripition of a movie watched by the user
    movies_desc : series
        movies descriptions from the movie database
    Returns
    -------
    movie_similarity_scores : list
        a list with similarity scores compared with user_movie_desc
    """

    movie_similarity_scores = []
    user = model(user_movie_desc)
    for movie in movies_desc.values:
        score = model(movie).similarity(user)
        movie_similarity_scores.append(score)
    return movie_similarity_scores

def main():
    """
    runs main program and print out suggested movies for user to watch based of movies
    which is most similar.

    Parameters
    ----------
    None

    Returns
    ----------
    watch_next: str
        print out the suggested movies for user to watch
    """
    movies_df = get_movie_df()
    nlp = spacy.load('en_core_web_md')
    score = recsys(nlp, user_data['Planet Hulk'], movies_df['Description'])
    movies_df['Similarity Score'] = score
    watch_next = movies_df.sort_values(by= 'Similarity Score', ascending= False,ignore_index=True)
    watch_next = watch_next['Movie Title'][0]
    print(f'Suggested Movie for User to watch next, is: {watch_next}')

if __name__ == '__main__':
    main()

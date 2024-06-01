"""Workout schema module"""
import graphene
from exercises.schema import ExerciseType
from exercises.models import Exercise


class Query(graphene.ObjectType):
    """Workout query class"""
    workout = graphene.List(ExerciseType,
                            body_part=graphene.String(),
                            exercise_name=graphene.String(),
                            equipment=graphene.String(),
                            level=graphene.String())

    def resolve_workout(self, info, **kwargs):
        """query resolver for workout property"""
        all_exercises = Exercise.objects.all()

        if kwargs.get('body_part'):
            all_exercises = all_exercises.select_related('body_part').filter(
                body_part__name=kwargs.get('body_part').lower())

        if kwargs.get('level'):
            all_exercises = all_exercises.select_related('level').filter(
                level__difficulty=kwargs.get('level').lower())

        if kwargs.get('exercise_name'):
            all_exercises = all_exercises.filter(
                name__icontains=kwargs.get('exercise_name').lower())

        if kwargs.get('equipment'):
            all_exercises = all_exercises.select_related('equipment').filter(
                equipment__name=kwargs.get('equipment').lower())

        return all_exercises

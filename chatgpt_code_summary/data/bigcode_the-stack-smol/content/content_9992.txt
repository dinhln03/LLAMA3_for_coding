from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from blog import hashing, models, schemas


def create(request: schemas.User, db: Session):
    new_user = models.User(name=request.name,
                           email=request.email,
                           password=hashing.Hash.bcrypt(request.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def get_one(id: int, db: Session):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'User with the id {id} is not found.')
    return user


def bulk_load(data, db: Session):

    for i in data:
        new_post = models.User(name=i[0],
                               email=i[1],
                               password=hashing.Hash.bcrypt(i[2]))
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
    return len(data)

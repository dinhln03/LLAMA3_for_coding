from typing import List, Optional

from discord.ext import commands
from fastapi import APIRouter
from pydantic import BaseModel

from bot.utils import ConnectionUtil, get_conn


class PartialAnswer(BaseModel):
    """Represents the model for an incomplete 8ball answer."""
    response: Optional[str]
    weight: Optional[int]


class Answer(PartialAnswer):
    """Represents an already inserted 8ball answer, is usually returned."""
    id: int


api = APIRouter(prefix='/8ball')


@api.get('/')
async def get_random_response(conn: ConnectionUtil = get_conn) -> str:
    # Weighted random selection
    return await conn.fetchval('SELECT response FROM eightball ORDER BY RANDOM()*weight;')


@api.get('/answers', response_model=List[Answer])
async def get_answers(conn: ConnectionUtil = get_conn) -> List[Answer]:
    # FastAPI converts it into a list of Answers
    return [record for record in await conn.fetch('SELECT * FROM eightball;')]


@api.post('/answers', response_model=Answer)
async def add_answer(answer: PartialAnswer, conn: ConnectionUtil = get_conn) -> Answer:
    # FastAPI handles convering it into an Answer
    return await conn.fetchrow(
        'INSERT INTO eightball (response, weight) VALUES ($1, $2) RETURNING *;',
        answer.response, answer.weight
    )


@api.get('/answers/{answer_id}', response_model=Answer)
async def get_answer(answer_id: int, conn: ConnectionUtil = get_conn) -> Answer:
    return await conn.fetchrow(
        'SELECT id, response, weight FROM eightball WHERE id=$1', answer_id
    )


@api.patch('/answers/{answer_id}', response_model=Answer)
async def edit_answer(
    answer_id: int, answer: PartialAnswer, conn: ConnectionUtil = get_conn
) -> Answer:
    return await conn.fetchrow("""
        UPDATE eightball SET
            response=COALESCE($2, response),
            weight=COALESCE($3, weight)
        WHERE id=$1 RETURNING *;
    """, answer_id, answer.response, answer.weight)


@api.delete('/answers/{answer_id}', response_model=Answer)
async def delete_answer(answer_id: int, conn: ConnectionUtil = get_conn) -> Answer:
    return await conn.fetchrow('DELETE FROM eightball WHERE id=$1 RETURNING *;', answer_id)


@commands.command(name='8ball')
async def eightball(ctx, *, question: str = None):
    await ctx.send(await get_random_response(ctx))

from dataclasses import dataclass, field
from datetime import datetime  # for typehinting
from typing import TYPE_CHECKING, Generator, List, Literal, Optional

import aiohttp
import dateparser

from .exceptions import UnsupportedRegionError
from .pricing import PriceQuery, query_price


COUNT = 30  # Items per page of paginated response

if TYPE_CHECKING:
    from .regions import Region  # pragma: no cover


@dataclass
class RatingContent:
    id: int = None
    name: str = None
    type: Literal["descriptor", "interactive"] = None
    image_url: Optional[str] = None  # JP Field
    svg_image_url: Optional[str] = None  # JP Field

    def __init__(self, data) -> None:
        self.id = data['id']
        self.name = data['name']
        self.type = data['type']
        if data.get('image_url'):
            self.image_url = data['image_url']
        if data.get('svg_image_url'):
            self.svg_image_url = data['svg_image_url']


@dataclass
class Rating:
    age: int = None
    id: int = None
    image_url: Optional[str] = None
    name: str = None
    provisional: bool = None
    svg_image_url: str = None

    def __init__(self, data) -> None:
        if (data['id']) == 0:
            return

        self.age = data['age']
        self.id = data['id']
        if data.get('image_url'):
            self.image_url = data['image_url']
        self.provisional = data['provisional']
        self.svg_image_url = data['svg_image_url']


@dataclass
class RatingSystem:
    id: int = None
    name: str = None

    def __init__(self, data) -> None:
        self.id = data['id']
        self.name = data['name']


@dataclass
class Game:
    region: "Region" = None
    content_type: str = None  # Literal["game", "bundle"] ??? expand and replace hint
    dominant_colors: List[str] = None
    formal_name: str = None
    hero_banner_url: str = None
    id: int = None
    is_new: bool = None
    membership_required: bool = None
    public_status: Literal["public"] = None
    rating_content: List[RatingContent] = field(default_factory=list)
    rating: Rating = None
    rating_system: RatingSystem = None
    release_date_on_eshop: datetime = None
    screenshots: List[str] = field(default_factory=list)
    strong_disclaimer: str = None
    tags: List = field(default_factory=list)
    target_titles: List = field(default_factory=list)

    def __init__(self, data, region) -> None:
        self.region = region
        self.content_type = data['content_type']
        self.dominant_colors = data['dominant_colors']
        self.formal_name = data['formal_name']
        self.hero_banner_url = data['hero_banner_url']
        self.id = data['id']
        self.is_new = data['is_new']
        self.membership_required = data['membership_required']
        self.public_status = data['public_status']
        self.rating_content = [RatingContent(c) for c in data['rating_info']['content_descriptors']]
        self.rating = Rating(data['rating_info']['rating'])
        self.rating_system = RatingSystem(data['rating_info']['rating_system'])
        # TODO: is this dateparser correct?
        self.release_date_on_eshop = dateparser.parse(data['release_date_on_eshop'], settings={'TIMEZONE': "UTC"})
        self.screenshots = [s['images'][0]['url'] for s in data['screenshots']]
        self.strong_disclaimer = data.get('strong_disclaimer', None)
        self.tags = data['tags']
        self.target_titles = data['target_titles']

    async def query_price(self) -> PriceQuery:
        return await query_price(self.region, self)


async def query_listing(region: "Region", type: Literal["sales", "new", "ranking"]) -> Generator[Game, None, None]:
    if not region.supports_listing:
        raise UnsupportedRegionError("Region does not support listings")

    if type not in ["sales", "new", "ranking"]:
        raise ValueError("Invalid type: " + type)

    lang, reg = region.culture_code.split('_')
    offset = 0

    async with aiohttp.ClientSession() as session:
        while True:
            url = f'https://ec.nintendo.com/api/{reg}/{lang}/search/{type}?offset={offset}&count={COUNT}'

            async with session.get(url) as request:
                request.raise_for_status()
                data = await request.json()

                for game in data['contents']:
                    yield Game(game, region)

                if (offset + COUNT) >= data['total']:
                    break

            offset += COUNT

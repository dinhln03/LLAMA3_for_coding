import asyncio
from typing import List

from app.common import SkipListing
from app.scrapers.base import BaseScraper


class MaartenScraper(BaseScraper):

    MAKELAARDIJ: str = "maarten"
    BASE_URL: str = "https://www.maartenmakelaardij.nl"

    # Specific functions
    async def extract_object_urls(self, soup) -> List[str]:
        """
        Extract apartment object urls
        """
        items = soup.find_all("a")
        urls: List[str] = []
        for item in items:
            if "woning/rotterdam-" in item["href"]:
                urls.append(item["href"])

        return list(set(urls))

    async def get_page_url(self, page_num: int) -> str:
        """
        Format page url
        """
        return f"{self.BASE_URL}/aanbod/rotterdam/"

    async def get_apartment_urls(self) -> List[str]:
        """
        Fetch list of apartment urls from inventory
        """
        urls = await self.scrape_page(0)
        return urls

    def extract_features(self, soup):
        """
        Extract feature metadata from listing
        """
        meta_data = {
            "makelaardij": self.MAKELAARDIJ,
            "building": {},
            "unit": {"energy": {}, "tags": []},
        }

        dt = soup.find_all("dt")
        dd = soup.find_all("dd")

        # Features
        for ind, key in enumerate(dt):

            if "Bouwjaar" in key.string:
                meta_data["building"]["year_constructed"] = self.find_int(
                    dd[ind].string
                )

            elif "Woonoppervlakte" in key.string:
                meta_data["unit"]["area"] = self.find_float(dd[ind].text.split(" ")[0])

            elif "Aantal kamers" in key.string:
                meta_data["unit"]["num_rooms"] = self.find_int(dd[ind].text)

            elif "verdiepingen" in key.string:
                meta_data["unit"]["num_floors"] = self.find_int(dd[ind].text)

            elif "Status" in key.string:
                meta_data["available"] = "Beschikbaar" in dd[ind].text

            elif "Buitenruimte" in key.string and "TUIN" in dd[ind].text:
                meta_data["unit"]["tags"].append("garden")

        # Other fields
        meta_data["address"] = soup.find("span", {"class": "adres"}).string
        meta_data["asking_price"] = self.find_int(
            soup.find("span", {"class": "price"}).string.replace(".", "")
        )

        description = soup.find("div", {"id": "read-more-content"}).children
        for p in description:
            p_text = str(p.text)
            if "Eigen grond" in p_text:
                meta_data["unit"]["own_land"] = True
            elif "erfpacht" in p_text:
                meta_data["unit"]["own_land"] = False

            if "Energielabel" in p_text:
                label = p_text.split("Energielabel: ")[1][0]
                meta_data["unit"]["energy"]["label"] = label

            break

        # Bounce broken listings
        if not meta_data["unit"].get("area"):
            raise SkipListing("Unable to find area")

        return meta_data


if __name__ == "__main__":
    scraper = MaartenScraper()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(scraper.start())

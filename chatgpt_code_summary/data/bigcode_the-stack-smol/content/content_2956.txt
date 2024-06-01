import time
import random
import os
import logging
import sys

from datetime import datetime
from datetime import timedelta
from urllib.parse import urlparse
from urllib.parse import urljoin

import click

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from pyvirtualdisplay import Display

from models import db
from models import Following
from models import Comment
from models import Like


username = os.environ['instagram_username']
password = os.environ['instagram_password']

dir_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    stream=sys.stdout,
    level='INFO',
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)

log = logging.getLogger('app')


def sleep(duration):
    log.info('Sleeping for {} seconds'.format(duration))
    time.sleep(duration)


def have_like(p):
    return random.randint(1, 100) < p


def get_url(driver):
    url = urlparse(driver.current_url)
    return urljoin('{}://{}'.format(url.scheme, url.netloc), url.path)


def get_driver(gui=True):
    options = webdriver.ChromeOptions()

    if not gui:
        options.add_argument('headless')
        options.add_argument('--no-sandbox')

    options.add_argument('window-size=1200x600')

    driver = webdriver.Chrome(
        executable_path='/usr/local/bin/chromedriver',
        chrome_options=options
    )

    driver.implicitly_wait(15)

    return driver


def login(driver, username, password):
    login_btn = driver.find_element_by_xpath("//p[@class='izU2O']/a[text()='Log in']")
    login_btn.click()
    sleep(5)

    login_input = driver.find_element_by_xpath("//INPUT[@name='username']")
    login_input.send_keys(username)

    password_input = driver.find_element_by_xpath("//INPUT[@type='password']")
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)

    sleep(10)


def search(driver, tag):
    driver.get('https://www.instagram.com/explore/tags/{tag}/'.format(tag=tag))
    sleep(4)

    first_image = driver.find_element_by_xpath(
        "//article/div[2]/div[1]/div[1]/div[1]"
    )
    first_image.click()

    sleep(2)


def go_to_next_photo(driver):
    try:
        nex_btn = driver.find_element_by_xpath(
            "//a[contains(@class, coreSpriteRightPaginationArrow)][text()='Next']"
        )
    except Exception:
        driver.save_screenshot('screenshot.png')
    else:
        nex_btn.click()

    time.sleep(1)


def is_already_liked(driver):
    try:
        driver.find_element_by_xpath("//span[@aria-label='Like']")
    except NoSuchElementException:
        log.info('Picture has already been liked {}'.format(driver.current_url))
        return True
    else:
        log.info('Picture has NOT been liked yet {}'.format(driver.current_url))
        return False


def like_post(driver):
    url = get_url(driver)
    try:
        Like.select().where(Like.url == url).get()
    except Like.DoesNotExist:
        pass
    else:
        log.info('Post has already been liked {url}'.format(url=url))
        return False

    try:
        like_btn = driver.find_element_by_xpath("//span[@aria-label='Like']")
    except NoSuchElementException:
        log.info('Could not find like button {}'.format(driver.current_url))
        time.sleep(1)

        return False
    else:
        log.info('Found like button. Trying to like {}'.format(driver.current_url))
        like_btn.click()

        Like.create(url=url)

    log.info('Liked picture {url}'.format(url=url))

    return True


def comment_post(driver, text):
    url = get_url(driver)
    try:
        Comment.select().where(Comment.url == url).get()
    except Comment.DoesNotExist:
        pass
    else:
        log.info('Post has already been commented {url}'.format(url=url))
        return False

    try:
        comment_input = driver.find_element_by_xpath('//TEXTAREA[@placeholder="Add a comment…"]')
    except NoSuchElementException as e:
        log.info(e)
        return False
    else:
        # comment_input.click()
        # comment_input.clear()
        # time.sleep(1)
        # comment_input = driver.find_element_by_xpath('//TEXTAREA[@placeholder="Add a comment…"]')

        # --------------------
        driver.execute_script(
            "arguments[0].value = '{} ';".format(text), comment_input
        )
        # An extra space is added here and then deleted.
        # This forces the input box to update the reactJS core
        comment_input.send_keys("\b")
        comment_input = driver.find_element_by_xpath('//TEXTAREA[@placeholder="Add a comment…"]')
        comment_input.submit()
        # --------------------

        # comment_input.send_keys(text)
        # comment_input.send_keys(Keys.RETURN)
        # comment_input.clear()

        Comment.create(url=url, comment=text)

    log.info('Commented picture {url} with "{text}"'.format(url=url, text=text))

    time.sleep(1)
    return True


def subscribe(driver):
    name_label = driver.find_element_by_xpath("//article/header//div[@class='e1e1d']/a[text()]")
    name = name_label.text

    follow_btn = driver.find_element_by_xpath("//article/header/div//button[text()]")

    try:
        following = Following.select().where(Following.name == name).get()
    except Following.DoesNotExist:
        pass
    else:
        log.info(
            'Already subscribed on user: @{user} ({following})'.format(
                user=name,
                following=following
            )
        )
        return False

    btn_text = follow_btn.text

    if btn_text == 'Follow':
        log.info('Going to subscribe on user: @{user}'.format(user=name))

        try:
            follow_btn.click()
            time.sleep(1)
        except Exception as e:
            log.info(e)
        else:
            Following.create(name=name)
            return True
    else:
        log.info('Already subscribed on user: @{user}'.format(user=name))
        return False


def get_random_comment():
    comments = [
        'Nice',
        'Nice photo',
        'Nice picture',
        'Nice capture',
        'Nice image',
        'Nice shot',
        'Great photo',
        'Great job',
        'Awesome picture',
        'awesome shot',
        'Like it',
        'Like this picture',
        'Like this photo',
        'Like this image',
        'Beautiful',
        'Beautiful photo',
        'Beautiful picture',
        'Lovely picture',
        'Lovely photo',
        'Amazing',
        'Amazing shot',
        'Amazing capture',
        'Amazing photo',
        'Wonderful shot',
        'Wonderful picture',
        'Wonderful photo',
    ]

    return random.choice(comments)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--tag', default='landscape', help='Instagram tag')
@click.option('--count', default=100, help='Number of user to follow')
@click.option('--gui/--no-gui', default=True, help='GUI')
def run_follower(tag, count, gui):
    driver = get_driver(gui)

    driver.get("https://www.instagram.com/")

    login(driver, username=username, password=password)

    search(driver, tag=tag)

    liked = 0
    commented = 0
    subscribed = 0

    while liked < count:
        go_to_next_photo(driver)

        was_liked = like_post(driver)

        if was_liked:
            liked += 1

        # if have_like(15) and comment_post(driver, text=get_random_comment()):
        # if comment_post(driver, text=get_random_comment()):
        #     commented += 1

        if have_like(33) and subscribe(driver):
            subscribed += 1

        log.info('Liked: {}, Commented: {} Subscribed {}'.format(liked, commented, subscribed))

        if was_liked:
            duration = random.randint(20, 60)
            sleep(duration)
        else:
            duration = random.randint(1, 8)
            sleep(duration)

    driver.close()


@cli.command()
@click.option('--count', default=100, help='Number of user to follow')
@click.option('--gui/--no-gui', default=True, help='GUI')
def run_unfollower(count, gui):
    initial_count = count

    driver = get_driver(gui)
    driver.implicitly_wait(3)

    driver.get("https://www.instagram.com/")

    login(driver, username=username, password=password)

    following_users = (
        Following.select()
        .where(
            Following.is_following == True,
            Following.date_created < datetime.now() - timedelta(days=14)
        )
        .order_by(Following.date_created)
    )
    for following in following_users:
        if count <= 0:
            return

        log.info(
            'Going to unfollow `@{user}` ({date})'.format(
                user=following.name, date=following.date_created
            )
        )

        driver.get("https://www.instagram.com/{name}".format(name=following.name))
        time.sleep(1)

        try:
            unfollow_btn = driver.find_element_by_xpath("//button[text()='Following']")
        except NoSuchElementException:
            still_following = False

            log.info('Already not following user `@{user}`'.format(user=following.name))

            following.is_following = False
            following.save()
        else:
            log.info('Still following user `@{user}`'.format(user=following.name))
            still_following = True
            unfollow_btn.click()
            duration = random.randint(5, 10)
            sleep(duration)

            try:
                unfollow_btn = driver.find_element_by_xpath(
                    "//div[@class='piCib']//button[text()='Unfollow']"
                    )
            except NoSuchElementException:
                pass
            else:
                still_following = True
                unfollow_btn.click()
                sleep(2)

        tries = 0
        while still_following:
            driver.refresh()
            try:
                driver.find_element_by_xpath("//button[text()='Follow']")
            except NoSuchElementException:
                pass
            else:
                still_following = False
                count -= 1
            try:
                driver.find_element_by_xpath("//button[text()='Follow Back']")
            except NoSuchElementException:
                pass
            else:
                still_following = False
                count -= 1

            if still_following:
                try:
                    unfollow_btn = driver.find_element_by_xpath("//button[text()='Following']")
                except NoSuchElementException:
                    pass
                else:
                    log.info(
                        'Still following user `@{user}` (tries {tries})'.format(
                            user=following.name,
                            tries=tries
                        )
                    )
                    still_following = True
                    unfollow_btn.click()
                    if tries == 0:
                        break

            tries += 1

        log.info('-- {count} of {initial_count} users are unfollowed --'.format(
            count=initial_count - count, initial_count=initial_count
        ))

    driver.close()


@cli.command()
def init_db():
    db.connect()
    db.create_tables([Following, Comment, Like])


if __name__ == "__main__":
    cli()

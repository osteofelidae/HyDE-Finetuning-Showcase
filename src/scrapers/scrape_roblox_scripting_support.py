# === DESCRIPTION ======================================================================================================
"""
Scrapes https://devforum.roblox.com/c/help-and-feedback/scripting-support/55
"""

# === DEPENDENCIES =====================================================================================================
from selenium import webdriver
from selenium.common import ElementClickInterceptedException, ElementNotInteractableException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from webdriver_manager.chrome import ChromeDriverManager
import time
from src.utils.path_utils import path
from pathlib import Path
from markdownify import markdownify
import json
import os


# === CONSTANTS ========================================================================================================
INDEX_URL = "https://devforum.roblox.com/c/help-and-feedback/scripting-support"
PAGE_LOAD_DELAY = 2
OUTPUT_DIR = path("data/roblox_forums_dumps")

SCROLL_STEPS = 500

SAVE_BATCH_SIZE = 3


# === FUNCTIONS ========================================================================================================
def get_solved_urls(
        driver: webdriver,
        url: str = INDEX_URL,
        scroll_steps: int = SCROLL_STEPS,
        page_load_delay: int = PAGE_LOAD_DELAY
):
    """
    Gets URLs of solved questions
    :param driver: Selenium webdriver.
    :param url: Base URL to scrape from.
    :param scroll_steps: Number of times to scroll down.
    :param page_load_delay: Delay to allow page to load.
    :return: list[str]
    """

    # Get page
    driver.get(url)
    time.sleep(page_load_delay)

    # Scroll down
    for _ in range(scroll_steps):
        scroll_height = 5000
        document_height_before = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script(f"window.scrollTo(0, {document_height_before + scroll_height});")
        time.sleep(page_load_delay)
        #while len(driver.find_elements(By.CSS_SELECTOR, "div.spinner")) > 0:
        #    pass

    # Get all topic list items
    topic_list_items = driver.find_elements(By.CSS_SELECTOR, "tr.topic-list-item")

    # Filter topic list items that are solved
    solved = [item for item in topic_list_items if len(item.find_elements(By.CSS_SELECTOR, "span.topic-status")) > 0]

    # Get links
    urls = [item.find_element(By.CSS_SELECTOR, "a.raw-topic-link").get_attribute("href") for item in solved]

    # Return result
    return urls

def get_conversation(
        driver: webdriver,
        url: str,
        page_load_delay: int = PAGE_LOAD_DELAY
):
    """
    Get conversation between OP and correct answerer.
    :param driver: Selenium webdriver.
    :param url: URl of page.
    :param page_load_delay: Delay to allow page to load.
    :return: str, list[dict[str:str]]
    """

    # Load url
    driver.get(url)
    time.sleep(page_load_delay)

    # Scroll down
    while True:
        scroll_height = 2000
        document_height_before = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script(f"window.scrollTo(0, {document_height_before + scroll_height});")
        time.sleep(2)
        document_height_after = driver.execute_script("return document.documentElement.scrollHeight")
        if document_height_after == document_height_before:
            break

    # Get all posts
    posts = driver.find_elements(By.CSS_SELECTOR, "div.topic-post")

    # Find title
    title = driver.find_element(By.CSS_SELECTOR, "a.fancy-title").get_attribute("innerHTML").strip()

    # Find initial post and author
    initial_post = posts[0]
    initial_author = initial_post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute("data-user-card")

    # Find solution post and author
    solution_post = [post for post in posts if len(post.find_elements(By.CSS_SELECTOR, "svg.d-icon-check")) > 0][0]
    solution_author = solution_post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute("data-user-card")

    if initial_author != solution_author:
        # Filter posts to include only posts between these two
        posts = [post for post in posts if post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute("data-user-card") == initial_author or post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute("data-user-card") == solution_author]

        # Get post content
        post_contents = []
        for post in posts:
            if post == solution_post:
                post_contents.append({
                    "name": "bot" if post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute(
                        "data-user-card") == initial_author else "bot",
                    "message": markdownify(post.find_element(By.CSS_SELECTOR, "div.cooked").get_attribute("innerHTML"))
                })
                break

            post_contents.append({
                "name": "user" if post.find_element(By.CSS_SELECTOR, "a.trigger-user-card").get_attribute("data-user-card") == initial_author else "bot",
                "message": markdownify(post.find_element(By.CSS_SELECTOR, "div.cooked").get_attribute("innerHTML"))
            })


        return title, post_contents

    else:
        return None, None


def get_conversations(
        driver: webdriver,
        urls: list[str],
        page_load_delay: int = PAGE_LOAD_DELAY
):
    """
    Get conversations from pages
    :param driver: Selenium webdriver.
    :param urls: List of urls.
    :param page_load_delay: Delay to allow page to load.
    :return: TODO
    """

    # Get conversations
    conversations = []
    for url in urls:
        try:
            title, history = get_conversation(
                driver=driver,
                url=url,
                page_load_delay=page_load_delay
            )
            if title is not None:
                conversations.append({
                    "title": title,
                    "history": history
                })
        except:
            print('failed')

    # Return result
    return conversations


# === MAIN =============================================================================================================
if __name__ == "__main__":

    # Selenium options
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Selenium driver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    # If urls file not found
    if not os.path.isfile(OUTPUT_DIR/"urls.txt"):

        # Get urls
        urls = get_solved_urls(driver=driver)[1:]

        # Write solved urls to file
        with open(OUTPUT_DIR/"urls.txt", "w") as file:
            file.writelines([url+"\n" for url in urls])

    # Get urls
    with open(OUTPUT_DIR/"urls.txt", "r") as file:

        urls = [line.strip() for line in file.readlines()]

    # Write to file
    with open(OUTPUT_DIR / "data.json", "w") as file:
        file.write("")

    # Get conversations; save every n
    for i in range(0, len(urls), SAVE_BATCH_SIZE):
        print(1.0 * i / len(urls) * 100)
        conversations = get_conversations(driver=driver, urls=urls[i:i+SAVE_BATCH_SIZE])

        # Write to file
        with open(OUTPUT_DIR/"data.json", "a") as file:
            file.writelines([json.dumps(conversation, indent=4)+"\n" for conversation in conversations])

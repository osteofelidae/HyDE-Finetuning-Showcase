# === DESCRIPTION ======================================================================================================
"""
Scrapes https://create.roblox.com/docs/scripting
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
INDEX_URL = "https://create.roblox.com/docs/scripting"
PAGE_LOAD_DELAY = 1
OUTPUT_DIR = path("data/roblox_docs_dumps")


# === FUNCTIONS ========================================================================================================
def expand_section(
    element: WebElement
):
    """
    Expands all list sections within element
    :param element: Element containing items to expand
    :return:
    """

    # Select section
    closed = [li for li in element.find_elements(By.CSS_SELECTOR, "li") if li.get_attribute("aria-expanded") == "false"]

    # Click all li
    for li in closed:

        try:
            li.click()
        except ElementClickInterceptedException:
            print(f"=== CLICK INTERCEPTED ===\n{li.get_attribute('outerHTML')}\n")
            time.sleep(1)
            li.click()
        except ElementNotInteractableException:
            print(f"=== NOT INTERACTABLE ===\n{li.get_attribute('outerHTML')}\n")
            time.sleep(1)
            li.click()

        # Expand sections in li
        expand_section(
            element=li
        )


def get_page_urls(
    driver: webdriver,
    index_url: str = INDEX_URL,
    page_load_delay: int = PAGE_LOAD_DELAY
):
    """
    Gets urls to scrape from
    :param driver: Selenium web driver.
    :param index_url: URL of index page
    :param page_load_delay: How much to wait (in seconds) when loading a page
    :return: list[str]
    """

    # Get page
    driver.get(index_url)
    time.sleep(page_load_delay)

    # Select section
    section = driver.find_elements(By.CSS_SELECTOR, "div.web-blox-css-mui-mix55w")[1]

    # Expand all sections
    expand_section(
        element=section
    )

    # Get URLs
    links = section.find_elements(By.CSS_SELECTOR, "a.MuiTypography-root")
    urls = [link.get_attribute("href") for link in links]

    # Return
    return urls


def dump_page_content(
    driver: webdriver,
    url: str,
    output_dir: Path = OUTPUT_DIR,
    page_load_delay: int = PAGE_LOAD_DELAY,
):
    """
    Dump page content as md into a file
    :param driver: Selenium webdriver
    :param url: URl of page
    :param output_dir: Output dir path
    :param page_load_delay: How much to wait (in seconds) when loading a page
    :return: dict[str:str]
    """

    # Load page
    driver.get(url)
    time.sleep(page_load_delay)

    # Select page content & title
    try:
        main_content = driver.find_element(By.CSS_SELECTOR, "article.web-blox-css-mui-mix55w")
        contents = driver.find_element(By.CSS_SELECTOR, "div.web-blox-css-mui-g7ht58").text.replace("\n", "; ")
        title = driver.find_element(By.CSS_SELECTOR, "h1.web-blox-css-mui-clml2g").get_attribute("innerHTML").replace("/", "_")
    except:
        try:
            main_content = driver.find_element(By.CSS_SELECTOR, "article.web-blox-css-mui-mix55w")
            contents = ""
            title = driver.find_element(By.CSS_SELECTOR, "h1.web-blox-css-mui-clml2g").get_attribute("innerHTML").replace("/", "_")
        except:
            print(f"ERROR LOADING PAGE: {url}")
            return

    # Get markdown content
    md = markdownify(main_content.get_attribute("innerHTML"))

    # Dump to file
    with open(output_dir/f"{title}.md", "w") as file:
        file.write(md)

    # Return params
    return {
        "title": title,
        "contents": contents
    }


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

    #  If URLs file does not exist
    if not os.path.isfile(OUTPUT_DIR/"urls.txt"):

        # Get page URLs
        urls = get_page_urls(
            driver=driver
        )

        # Write lines to file
        with open(OUTPUT_DIR/"urls.txt", "w") as file:
            file.writelines([f"{url}\n" for url in urls])

    # Get URLs from file
    with open(OUTPUT_DIR / "urls.txt", "r") as file:
        urls = [url.strip() for url in file.readlines()]

    # Dump content
    contents = []
    for url in urls:
        contents.append(dump_page_content(
            driver=driver,
            url=url
        ))

    # Save contents
    with open(OUTPUT_DIR/"contents.json", "w") as file:
        json.dump(
            obj={
                "contents": contents
            },
            fp=file
        )

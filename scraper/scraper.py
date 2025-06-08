# scraper.py
from live_scraping import scrape_kompas_page_1, scrape_detik_articles_with_categories, scrape_cnn_today, scrape_liputan6_live

def scrape_all_sources():
    """
    Menggabungkan semua data dari berbagai sumber.
    """
    kompas_data = scrape_kompas_page_1()
    detik_data = scrape_detik_articles_with_categories("https://www.detik.com/tag/bencana-alam/?sortby=time&page=1")
    cnn_data = scrape_cnn_today()
    liputan6_data = scrape_liputan6_live()

    return kompas_data + detik_data + cnn_data + liputan6_data

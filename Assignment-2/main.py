import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_range():
    for i in range(1, 137, 9):
        url_range = str(i) + "-" + str(i + 9)
        yield url_range


def encode_images(images, maximum=3):
    while len(images) < maximum:
        images.append(None)
    return images[:maximum]  # in case we have more ...


def create_list_of_sizes():
    deti = [68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164, 170, 176]
    zeny = [42, 44, 46, 48, 50]
    muzi = [46, 48, 50, 52, 54]
    all = deti
    all.extend(zeny)
    all.extend(muzi)
    all = sorted(list(set(all)))
    all_str = []
    for i in all:
        all_str.append(str(i))
    print(all_str)


def determine_type(sizes):
    sizes_only = [i[0] for i in sizes]
    if 54 in sizes_only:
        return "M"
    elif 50 in sizes_only:
        return "W"
    else:
        return "K"


def encode_sizes(sizes):
    str_sizes = ['42', '44', '46', '48', '50', '52', '54', '68', '74', '80', '86', '92', '98', '104', '110', '116', '122', '128', '134', '140', '146', '152', '158', '164', '170', '176']
    bool_array = []
    int_sizes = [i[0] for i in sizes]
    availability = [i[1] for i in sizes]

    for str_s in str_sizes:
        if int(str_s) in int_sizes:
            avail = availability[int_sizes.index(int(str_s))]
            bool_array.append(avail)
        else:
            bool_array.append(None)
    return bool_array


def scrape_details(website, detail_url):
    page = requests.get(detail_url)
    soup = BeautifulSoup(page.content, features="lxml")

    # Images ...
    images_block = soup.findAll("div", {"class": "prooductView__image"})[0]
    images = []
    for image_div in images_block.findAll("div", {"class": "main-image-nav-item"}):
        img_src = website + image_div.find_all('img', src=True)[0]['src']
        images.append(img_src)

    # Info about product ...
    info_block = soup.findAll("div", {"class": "prooductView__info"})[0]
    headline_text = info_block.findAll("div", {"class": "prooductView__headline"})[0].text
    price = info_block.findAll("div", {"class": "prooductView__price"})[0].text.replace("Цена:", "").replace("Р", "").strip()
    retail_price = info_block.findAll("div", {"class": "prooductView__roznprice"})[0].text.replace("Розничная цена", "").replace("Р", "").strip()
    product_text = info_block.findAll("div", {"class": "prooductView__desc"})[0].text

    # Sizes
    sizes = []
    for size_item in info_block.findAll("div", {"class": "product-item"}):
        size = size_item.findAll("div", {"class": "name"})[0].text.replace("рост", "").replace("Размер", "").replace("размер", "").strip()
        avail_text = size_item.findAll("div", {"class": "buy-button"})[0].text
        available = False if "нет на складе" in avail_text else True
        sizes.append((int(size), available))
    # ['url', 'name', 'type', 'price', 'price_r', 'text', 'img1', 'img2' 'img3', size1, size2, ....]
    type = determine_type(sizes)
    to_return = [detail_url, headline_text, type, float(price), float(retail_price), product_text]
    to_return.extend(encode_images(images, maximum=3))
    assert len(to_return) == 9
    to_return.extend(encode_sizes(sizes))

    return to_return


def parse_item(website, item_block):
    detail_url = website + item_block.findAll("div", {"class": "product__image"})[0].find_all('a', href=True)[0]['href']
    return scrape_details(website, detail_url)


def main():
    website = "https://xn--80apgqhlea0j.xn--p1ai"
    base_product_url = website + "/component/virtuemart/search/results,"

    cols = ['url', 'name', 'type', 'price', 'price_r', 'text', 'img1', 'img2', 'img3']
    sizes = ['42', '44', '46', '48', '50', '52', '54', '68', '74', '80', '86', '92', '98', '104', '110', '116', '122', '128', '134', '140', '146', '152', '158', '164', '170', '176']
    cols.extend(sizes)
    df = pd.DataFrame(columns=cols)

    glob_idx = 0
    all_goods = 145
    for end_url in get_range():
        current_url = base_product_url + end_url
        page = requests.get(current_url)
        soup = BeautifulSoup(page.content, features="lxml")
        product_list = soup.findAll("div", {"class": "products__list"})
        product_list = product_list[0]
        items = product_list.findAll("div", {"class": "product__item"})
        for item in items:
            parsed_item = parse_item(website, item)
            # [detail_url, headline_text, type, price, retail_price, product_text, img1, img2, img3, sizes...]
            df.loc[len(df)] = parsed_item
            glob_idx += 1
            print("{}/{}".format(glob_idx, all_goods))
        print("Range {} parsed".format(end_url))
        df.to_csv("eshop.csv", index=False)


if __name__ == '__main__':
    df = pd.read_csv("eshop.csv")
    df = df.fillna(0)
    df.to_csv("eshop2.csv", index=False)
    # df['material'] = df['text'].apply(lambda x: x.split("Состав")[1] if len(x.split("Состав")) > 1 else x.split("состав")[1])
    df['material'] = "50%шерсть50%ПАН(высокообъемный акрил аналог шерсти)"
    df['text'] = df['text'].apply(lambda x: x.split("Состав")[0] if len(x.split("Состав")) > 1 else x.split("состав")[0])
    df['text'] = df['text'].apply(lambda x: x.replace(".", "").strip())
    for i in sorted(df['text'].unique().tolist()):
        print(i)
    print("-------------------------------")
    for i in sorted(df['material'].unique().tolist()):
        print(i)
    df.to_csv("eshop2.csv", index=False)


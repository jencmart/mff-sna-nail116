import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_ingredients(recipe_link):
    page = requests.get(recipe_link)
    soup = BeautifulSoup(page.content, features="lxml")
    ingredients_div = soup.find("div", {"class": "tasty-recipes-ingredients-body"}).text
    ingredients_div = ingredients_div.lower().strip().splitlines()
    parsed_list = []

    units = ['teaspoon or less', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons', 'cup', 'cups', 'pint', 'pints',
             'quart', 'quarts', 'pecks', 'peck', 'dashes', 'dash', 'ounces', 'ounce', 'pound', 'pounds', 'oz',
             'gallons', 'gallon', 'less than', 'plus', 'large', 'etc', 'and', 'pinch' ]

    common_stopwords = ['fresh', 'cut', 'pound', 'into', 'chopped', 'large', 'for', 'diced', 'ounce', 'the', 'pounds',
                        'sliced', 'about', 'red', 'whole', 'pieces', 'small', 'peeled', 'see', 'white', 'divided',
                        'minced', 'can', 'half', 'medium', 'or', 'of', 'a', 'slices', 'shredded', 'plus', 'dry', 'used',
                        'ground', 'more', 'partially', 'our', 'how', 'to', 's', 'section', 'on', 'how', 'blind', "i",
                        "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                        "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                        "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                        "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
                        "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    other_stopwords = ["vidalia", "bell","black","italian","quartered","also","not","jack","boneles","finely","hot","frozen","favorite","spray","top","cold","kosher","purpose","thin","thigh","roll","one","confectioner","removed","unsalted","zest","inch","choice","ripe","available","leav","heavy","two","green","coarsely","mix","needed","skinles","good","sprig","drained","semi","dark","light","frying","sweet","can","stock","roasted","crumb","yellow","box","cooked","beaten","brown","note","stemmed","ingredient","strip","pinch","supermarket","skin","sized","left","virgin","long","optional","packed","grated","granulated","thinly","sea","grind","thick","cored","taste","organic","seeded","recipe","kitchen","rib","store","unsweetened","cooking","package","casing","homemade","bought","bottom","brand","sour","pie","cub","extra","crushed","uncooked","approximately","cleaned","dice","jarred","stick","softened","baking","liquid","quality","jar","golden","leftover","paste","crust","serve","use","fine","flat","squeezed","dried","extract","temperature","breast","freshly","flak","trimmed","room","bite","pizza","additional","using","head","cake","serving","stem","pan","melted","weight","stalk","bay","quarter","like","noodl","prepared","bag","raw","grater","canned","powder","hol","bunch","leaf","sharp",]

    remove = units.extend(common_stopwords)
    remove = remove.extend(other_stopwords)

    for ingredient in ingredients_div:
        ingredient = re.sub('[^a-z]+', ' ', ingredient)
        ingredient = ingredient.strip()
        clean = [token.lower().strip() for token in ingredient.strip().split() if
                 (token not in remove)  and len(token) > 2]
        clean = [w if w[-2:] != 'es' else w[:-2] for w in clean]
        clean = [w if w[-1:] != 's' else w[:-1] for w in clean]
        clean = [w for w in clean if len(w) > 2]
        clean = " ".join(clean)
        parsed_list.append(clean)
    parsed_list = ",".join(parsed_list)
    return parsed_list


def download_data(fname):
    base_url = "https://www.afamilyfeast.com/recipe-index/page/"

    all_res = []
    glob_idx = 0
    all_goods = 105 * 16 + 3
    for end_url in range(1, 106):
        end_url = str(end_url)
        current_url = base_url + end_url
        page = requests.get(current_url)
        soup = BeautifulSoup(page.content, features="lxml")

        recipes = soup.findAll("div", {"class": "grid-info"})

        for recipe in recipes:
            # parse recipes only...
            if len(recipe.findAll("i", {"class": "icon-recipe"})) == 0:
                continue

            link = recipe.find("h2")
            recipe_name = link.text.replace(",", " ").strip()
            recipe_link = link.find("a", href=True)['href']
            ingredients = get_ingredients(recipe_link)
            result = recipe_name + ',' + ingredients
            all_res.append(result)
            glob_idx += 1
            print("{}/{}".format(glob_idx, all_goods))
            if glob_idx == 1000:
                print("DONE")

                with open(fname, 'w') as f:
                    for res in all_res:
                        f.write(res + "\n")
                return

        print("Range {} parsed".format(end_url))


class CountDict:
    def __init__(self):
        self._dict = {}

    def insert(self, key):
        if key in self._dict:
            self._dict[key] += 1
        else:
            self._dict[key] = 1

    def get_geq_limit(self, limit):
        result_set = set()
        for k, v in self._dict.items():
            if v >= limit:
                result_set.add(k)

        return result_set


def save_stop_words(file, target):

    common_ingredients = set()
    with open("common_ingredients.txt", 'r') as f:
        for w in f:
            common_ingredients.add(w.strip())

    counts = CountDict()

    stopwords1 = set()
    with open("stopwords.txt", 'r') as f:
        for w in f:
            stopwords1.add(w.strip())

    rare_words = set()
    with open("rare.txt", 'r') as f:
        for w in f:
            rare_words.add(w.strip())

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            ingredient_list = line.split(",")[1:]
            for ingredient in ingredient_list:
                for w in ingredient.split():
                    counts.insert(w)
    possible_stop_words = counts.get_geq_limit(limit=1)

    print(len(possible_stop_words))
    with open(target, 'w') as f:
        for w in possible_stop_words:
            if w not in common_ingredients and w not in stopwords1 and w not in rare_words:
                f.write(w + "\n")


def filter_stopwords_from_result(result_file, stopword_file, new_file):

    # Load stop-word list
    stopwords = set()
    with open(stopword_file, 'r') as f:
        for w in f:
            stopwords.add(w.strip())

    rare_words = set()
    with open("rare.txt", 'r') as f:
        for w in f:
            rare_words.add(w.strip())

    # filter our results...
    filtered_lines = []
    with open(result_file, 'r') as f:
        for line in f:
            line = line.strip()
            ingredient_list = line.split(",")[1:]
            new_ingredient_list = [line.split(",")[0]]
            for ingredient in ingredient_list:
                ingredient = [w for w in ingredient.split() if w not in stopwords and w not in rare_words]
                if len(ingredient) > 0:

                    ingredient  = sorted(list(set(ingredient)))
                    # if len(ingredient) <= 3:
                    #
                    #     ingredient = " ".join(ingredient)
                    #     new_ingredient_list.append(ingredient)
                    # else:
                    #     print(ingredient)
                    for i in ingredient:
                        new_ingredient_list.append(i)

            rr = sorted(list(set(new_ingredient_list)))
            filtered_lines.append(",".join(rr))

    # Save it ..
    with open(new_file, 'w') as f:
        for res in filtered_lines:
            f.write(res + "\n")
    return


if __name__ == '__main__':
    fname = "result.txt"

    # 1. download data
    # download_data(fname)

    # 2. save possible stop words
    # save_stop_words("result.txt", "stopwords-extra.txt")

    # 3. Filter stopwords
    filter_stopwords_from_result(fname, "stopwords.txt", "result-filtered.txt")



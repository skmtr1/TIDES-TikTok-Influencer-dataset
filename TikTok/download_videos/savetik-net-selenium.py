from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os
import traceback
import pickle

dir_base = os.path.abspath(os.curdir)
output_folder = os.path.join(dir_base, "videos")

chromeOptions = webdriver.ChromeOptions()
prefs = {"download.default_directory" : output_folder,
         "directory_upgrade": True,
         'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}
chromeOptions.add_experimental_option("prefs",prefs)
chromeOptions.add_argument("--headless=new")
chromeOptions.add_extension('gighmmpiobklfepjocnamgkkbiglidom.crx')  # chnahge the crx file path 
#chromeOptions.add_extension('/home/pier/.config/google-chrome/Default/Extensions/gighmmpiobklfepjocnamgkkbiglidom/5.6.0_0/')

chromedriver = "./chromedriver_win32/chromedriver.exe" # set the location of the browser drivers
driver = webdriver.Chrome(chromedriver, options=chromeOptions)

time.sleep(10)

videos = [line.rstrip() for line in open("f3.txt")]
videos.reverse()


try:
    already_done = pickle.load(open("already_done.pkl","rb"))
except:
    already_done = set()

print(already_done)

already_done_files = os.listdir(output_folder)
for a in already_done_files:
    print(a)
    if "-" in a:
        if "-(" in a:
            already_done.add(a.split("-")[1])
        else:
            already_done.add(a.split(" - ")[1].replace(".mp4",""))
    if "_" in a:
        already_done.add(a.split("_")[1].replace(".mp4",""))

print("already_done: ", already_done)

c = 0
for v in videos:
    v_id = v.split("/")[-1]
    if v_id in already_done:
        #print(f"{v_id} already exists!")
        continue

    try:
        print(f"downloading {v_id}...")
        driver.get("https://savetik.net/")
        time.sleep(3)
        search_bar = driver.find_element("name","url") #search bar
        search_bar.clear()
        search_bar.send_keys(v)
        search_bar.send_keys(Keys.RETURN)
        #time.sleep(5)
        #driver.find_element("xpath",'//a[normalize-space()="Download Server 01"]').click()

        elem = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.LINK_TEXT,"Download Video"))
        )
        
        time.sleep(1)
        elem.click()
        print("sleeping...")
    except Exception as e:
        print(f"error downloading {v_id}")
        traceback.print_exc()
        time.sleep(10)
    time.sleep(10)
    c += 1   
    if c % 50 == 0:
        already_done_files = os.listdir(output_folder)
        for a in already_done_files:
            if " - " in a:
                already_done.add(a.split(" - ")[1].replace(".mp4",""))
            if "_" in a:
                already_done.add(a.split("_")[1].replace(".mp4",""))


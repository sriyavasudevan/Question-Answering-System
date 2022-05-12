import re
import pandas as pd

ans_for_display = answer

df = pd.read_csv('map_of_hyperlinks.csv')

list = re.findall("link\d\d", ans_for_display)


for link in list:
    hyperlink = df.loc[df['Link'] == link]['Hyperlink'].item()
    ans_for_display = ans_for_display.replace(link, hyperlink)






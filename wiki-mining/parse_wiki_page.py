import mwparserfromhell

f = open('albert_einstein_wiki.txt', 'r')
ori_txt = f.read()
text = ori_txt.decode('utf8')

# parsed text from wiki text
parsed_text = mwparserfromhell.parse(text).strip_code()
#print parsed_text.encode("utf8")
from HTMLParser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.to_remove = False
        self.fed = []
    def handle_starttag(self, tag, attrs):
        if tag.upper() == 'REF':
            self.to_remove = True
    def handle_endtag(self, tag):
        if tag.upper() == 'REF':
            self.to_remove = False
    def handle_data(self, d):
        if not self.to_remove:
            self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()

clean_text = strip_tags(parsed_text)
print clean_text.encode("utf8")
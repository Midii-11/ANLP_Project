import wikipedia

person_page = wikipedia.page('Albert Einstein')

print(person_page.summary)
print('\n')
print(person_page.content)

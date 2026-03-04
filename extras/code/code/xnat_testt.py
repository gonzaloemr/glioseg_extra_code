import xnat


BIGR_server = "https://xnat-rad.research.erasmusmc.nl"

with xnat.connect(BIGR_server) as session:
    print(session.projects)

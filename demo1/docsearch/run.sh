docker pull solr

docker run -d -p 8983:8983 --name esg_search_solr solr solr-create -c esg_search_core

text_cleaner<-function(corpus){
  tempcorpus = lapply(corpus,toString)
  #for(i in 1:length(tempcorpus)){
  #  tempcorpus[[i]]<-iconv(tempcorpus[[i]], "ASCII", "UTF-8", sub="")
  #}
  tempcorpus = lapply(tempcorpus, tolower)
  tempcorpus<-Corpus(VectorSource(tempcorpus))
  toSpace <- content_transformer(function (x , pattern ) gsub(pattern, "", x))
  
  #Removing all the special charecters and words
  tempcorpus <- tm_map(tempcorpus, toSpace, "@")
  tempcorpus <- tm_map(tempcorpus, toSpace, "\\|")
  tempcorpus <- tm_map(tempcorpus, toSpace, "#")
  tempcorpus <- tm_map(tempcorpus, toSpace, "http")
  tempcorpus <- tm_map(tempcorpus, toSpace, "https")
  tempcorpus <- tm_map(tempcorpus, toSpace, ".com")
  tempcorpus <- tm_map(tempcorpus, toSpace, "$")
  tempcorpus <- tm_map(tempcorpus, toSpace, "+")
  tempcorpus <- tm_map(tempcorpus, removeNumbers)
  # Remove english common stopwords
  tempcorpus <- tm_map(tempcorpus, removeWords, stopwords("english"))
  # Remove punctuation
  tempcorpus <- tm_map(tempcorpus, removePunctuation)
  
  # Eliminate extra white spaces
  tempcorpus <- tm_map(tempcorpus, stripWhitespace)
  # Stem the document
  tempcorpus <- tm_map(tempcorpus, PlainTextDocument)
  tempcorpus <- tm_map(tempcorpus,  stemDocument, "english")
  # Remove uninformative high-frequency words
  tempcorpus <- tm_map(tempcorpus, toSpace, "amp")
  return(tempcorpus)
}

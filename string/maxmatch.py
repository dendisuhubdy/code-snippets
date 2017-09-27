"""
The Max Match algorithm

Pseudo code

function MaxMatch(sentence, dictionary D)

	if sentence is empty:
		return empty_list
	for i <- length(sentence) downto 1
		firstword = first i chars of sentence
		remainder = rest of sentence
	if InDictionary(firstword, D)
		return list(firstword, MaxMatch(remainder, dictionary))

returns word sequence W

"""


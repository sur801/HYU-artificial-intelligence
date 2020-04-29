from collections import defaultdict
from pandas import read_table
import numpy as np
import math
from konlpy.tag import Twitter

twitter = Twitter()

def tokenize(doc):
  # norm, stem은 optional
  return ['/'.join(t) for t in twitter.pos(doc)]
  #return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]
class NaiveBayesClassifier:

    def __init__(self, k=1.0):
        self.k = k
        self.word_probs = []

    def load_corpus(self, path):
    	#train 할 doc들을 읽어오는 함수
        corpus = []
        f = open(path, "r")
        # 맨 첫줄은 id, documnet, label 이라고 써있는 거니까 그냥 읽어서 버린다.
        f.readline()

        while  True:
        	line = f.readline()
        	if not line : break;
        	# tab으로 구분 되어 있으니 tab을 기준으로 다 잘라준다.
        	temp = line.split('\t')
        	val = temp[2][0]
        	temp.pop(2)
        	# 0번째는 다 그냥 id값이니 빼버린다.
        	temp.pop(0)
        	# readline으로 읽어온 후 split을 하면 label값이 string으로 바뀌 므로 float으로 바꾼 다음 다시 temp에 집어넣는다.
        	temp.append(float(val))
        	corpus.append(temp)
        f.close()
        return corpus

    def count_words(self, training_set):

        # 학습데이터는 영화리뷰 doccumnet , 긍정 부정 값(1, 0)으로 구성
        counts = defaultdict(lambda : [0, 0])
        cnt = 0
        for doc, point in training_set:

            # konlpy로 tokenize
            words = tokenize(doc)
            # 학습 데이터를 10000개의 배수로 읽을 때마다 몇개 읽었는지 출력.
            cnt +=1
            if cnt%10000 == 0 :
            	print(cnt) 
            for word in words:
                counts[word][0 if point == 1 else 1] += 1
        return counts



    def word_probabilities(self, counts, total_pos, total_neg, k):
        # 단어의 나오는 빈도수 를 [단어, p(w|긍정), p(w|부정)] 형태로 반환
        return [(w,
                 (pos + k) / (total_pos + 2*k),
                 (neg + k) / (total_neg + 2*k))
                for w, (pos, neg) in counts.items()]

    def compare_probability(self, word_probs, doc):
        #konlpy 이용해서 tokenize
        docwords = tokenize(doc)

        # 모두 0으로 초기화 해준다.
        log_prob_if_pos = log_prob_if_neg = 0.0

        # 모든 단어에 대해 반복해준다.
        for word, prob_if_pos, prob_if_neg in word_probs:
            # 만약 리뷰에 word가 있다면
            # 해당 단어가 나올 log 확률을 더해 줌
            if word in docwords:
                #print(word)
                log_prob_if_pos += math.log(prob_if_pos)
                log_prob_if_neg += math.log(prob_if_neg)

            # 만약 리뷰에 word가 없다면
            # 해당 단어가 없을 log 확률을 더해 줌
            else:
                log_prob_if_pos += math.log(1.0 - prob_if_pos)
                log_prob_if_neg += math.log(1.0 - prob_if_neg)

        prob_if_pos = math.exp(log_prob_if_pos)
        prob_if_neg = math.exp(log_prob_if_neg)
        # 이렇게 긍정 확률과 부정 확률을 구한 후 둘 중 더 큰 확률을 채택해서, 내가 읽은 review가 긍정인지 부정인지 결정을 해준다.
        return 1 if (prob_if_pos / (prob_if_pos + prob_if_neg)) > (prob_if_neg / (prob_if_pos + prob_if_neg)) else 0

    def train(self, trainfile_path):
        training_set = self.load_corpus(trainfile_path)

        # 범주0(긍정)과 범주1(부정) 문서 수를 세어 줌
        num_pos = len([1 for _, point in training_set if point == 1])
        num_neg = len(training_set) - num_pos

        # training!!!!
        word_counts = self.count_words(training_set)
        self.word_probs = self.word_probabilities(word_counts,
                                                  num_pos,
                                                  num_neg,
                                                  self.k)

    def classify(self, doc):
        return self.compare_probability(self.word_probs, doc) 

def resWrite(path):
	model = NaiveBayesClassifier()

	model.train(trainfile_path='ratings_train.txt')

	fi = open(path, "r")
	fo = open("ratings_result.txt", "w")

	first = fi.readline()
	fo.write(first)

	#true postive, true negative, false postive, false negative 계산
	# tp = 0
	# tn = 0
	# fp = 0
	# fn = 0

	while True :
		line = fi.readline()
		if not line : break;
		temp = line.split('\t')
		#valid data classify후 결과 비교 하기 위한 부분
		# label = int(temp.pop(2))
		res = model.classify(temp[1])

		#valid data classify후 결과 비교 하기 위한 부분
		# if(res == 1 and label == 1) : 
		# 	tp += 1.0
		# elif(res == 0 and label == 1) :
		# 	fn += 1.0
		# elif(res == 1 and label == 0) :
		# 	fp += 1.0
		# elif(res == 0 and label == 0):
		# 	tn += 1.0
		fo.write(temp[0] + '\t' + temp[1] + '\t' + str(res) + '\n')


	# classify한 결과 출력
	# print("precision : ", tp/(tp+fp))
	# print("recall : ", tp/(tp+fn))
	# print("false : ", int(fn+fp))

	fi.close()
	fo.close()




def main():
	#classify 한 결과를 써주기 위한 함수
	#path는 내가 classify할 데이터
    resWrite(path = 'ratings_test.txt')

    return 0

if __name__ == "__main__":
    main()

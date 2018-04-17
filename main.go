package main

import (
	"fmt"
	"strings"
)

// Class contains the document count, an array of all words used in those documents
// (the array contains duplicates) and a map with the word frequency which can
// be used to obtain the unique word count.
type Class struct {
	Documents float64
	Words     []string
	WordFreq  map[string]float64
}

// Classifier gives us total document count, the length of the NSplit being used,
// a map with all classes and a map with the word frequency which again can be
// used to obtain the unique word count.
type Classifier struct {
	NSplit      int
	Documents   float64
	Classes     map[string]*Class
	UniqueWords map[string]float64
}

// SplitWords returns an array of sequences of n items. The length n is defined
// by the parameter size.
//
// The input (1, "this outputs NSplit") would be ["this", "outputs", "NSplit"].
//
// The input (2, "this outputs NSplit") would be ["this outputs", "outputs NSplit"].
func SplitWords(size int, sentence string) []string {
	sliptedWords := []string{}
	words := strings.Split(sentence, " ")

	if len(words) <= size {
		sliptedWords = append(sliptedWords, strings.Join(words, " "))
		return sliptedWords
	}

	for i := 0; i+size <= len(words); i++ {
		sliptedWords = append(sliptedWords, strings.Join(words[i:i+size], " "))
	}

	return sliptedWords
}

// NewClassifier returns a new classifier which initiates two empty maps. This
// could later be improved so that everything is saved more efficiently.
func NewClassifier(n int) *Classifier {
	return &Classifier{
		NSplit:      n,
		Classes:     make(map[string]*Class),
		UniqueWords: make(map[string]float64),
	}
}

// Train adds the splitted words of a sentence to an existing or new class.
func (c *Classifier) Train(class string, sentence string) {
	c.Documents++
	_, exists := c.Classes[class]
	if exists == false {
		c.Classes[class] = &Class{
			WordFreq: make(map[string]float64),
		}
	}

	c.Classes[class].Documents++
	words := SplitWords(c.NSplit, sentence)
	for _, word := range words {
		c.UniqueWords[word]++
		c.Classes[class].Words = append(c.Classes[class].Words, word)
		c.Classes[class].WordFreq[word]++
	}
}

// GetPrior returns the prior probabilities of a document being in a specific
// class. It is calculated by dividing the class frequency by the total amount
// of documents.
func (c *Classifier) GetPrior(class string) float64 {
	return c.Classes[class].Documents / c.Documents
}

// Classify returns the probabilities for a sentence belonging to a
// certain class. These probabilities are calculated by taking the class prior
// P(class) and multiplying it by the conditional probabilities P(word|class).
func (c *Classifier) Classify(sentence string) map[string]float64 {
	uniqueWordCount := float64(len(c.UniqueWords))
	words := SplitWords(c.NSplit, sentence)
	cProbabilities := make(map[string]float64)

	for class, data := range c.Classes {
		prior := c.GetPrior(class)
		classWordCount := float64(len(data.Words))
		wProbabilities := make(map[string]float64)
		for _, word := range words {
			frequency, exists := data.WordFreq[word]
			if exists == false {
				frequency = 0
			}

			wProbabilities[word] = (frequency + 1.0) / (classWordCount + uniqueWordCount)
		}
		result := prior
		for _, value := range wProbabilities {
			result = result * (value)
		}
		cProbabilities[class] = result

	}

	return cProbabilities
}

func main() {
	classifier := NewClassifier(1)

	classifier.Train("bom", "eu te adoro")
	classifier.Train("bom", "eu te amo")
	classifier.Train("bom", "eu amo batatas fritas")
	classifier.Train("bom", "eu amo bolo")
	classifier.Train("bom", "voce é demais")
	classifier.Train("bom", "bolo que é demais")

	classifier.Train("ruim", "peixe é ruim")
	classifier.Train("ruim", "eu te odeio")
	classifier.Train("ruim", "eu quero ver queimar")
	classifier.Train("ruim", "eu quero é que se exploda")
	classifier.Train("ruim", "eu acho que isso é muito ruim")
	classifier.Train("ruim", "odeio ficar parado")

	probabilities := classifier.Classify("nao achei o filme ruim")
	// probabilities := classifier.Classify("eu gosto de bolo pra caramba")

	// probabilities := classifier.Classify("eu acho ruim voce assistir TV")
	// probabilities := classifier.Classify("eu odeio micro-ondas")

	if probabilities["bom"] > probabilities["ruim"] {
		fmt.Printf("bom")
	} else {
		fmt.Printf("ruim")
	}
}

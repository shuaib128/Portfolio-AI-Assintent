package main

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"math"
	"net/http"

	"github.com/gin-gonic/gin"
)

type KBItem struct {
	ID  string    `json:"id"`
	Q   string    `json:"q"`
	A   string    `json:"a"`
	Vec []float64 `json:"vec"`
}

type AskRequest struct {
	Q string `json:"q"`
}

type EmbeddingResp struct {
	Embedding []float64 `json:"embedding"`
}

var KB []KBItem

// cosine similarity
func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// embed user query via Ollama
func getEmbedding(text string) ([]float64, error) {
	payload := map[string]string{"model": "nomic-embed-text", "prompt": text}
	body, _ := json.Marshal(payload)

	resp, err := http.Post("http://localhost:11434/api/embeddings", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out EmbeddingResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Embedding, nil
}

func main() {
	// load your kb_embedded.json
	data, _ := ioutil.ReadFile("kb_embedded.json")
	json.Unmarshal(data, &KB)

	r := gin.Default()

	r.GET("", func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{"message": "Hey this is Sams personal assistance"})
	})

	r.POST("/ask", func(c *gin.Context) {
		var req AskRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "bad request"})
			return
		}

		qvec, err := getEmbedding(req.Q)
		if err != nil {
			c.JSON(500, gin.H{"error": "embedding failed"})
			return
		}

		// find best match
		best := KB[0]
		bestScore := -1.0
		for _, item := range KB {
			score := cosine(qvec, item.Vec)
			if score > bestScore {
				best = item
				bestScore = score
			}
		}

		if bestScore < 0.4 {
			c.JSON(200, gin.H{"text": "I don’t have that in my profile.", "score": bestScore})
			return
		}

		c.JSON(200, gin.H{
			"text":             best.A,
			"matched_question": best.Q,
			"score":            bestScore,
			"id":               best.ID,
		})
	})

	r.Run(":8080")
}

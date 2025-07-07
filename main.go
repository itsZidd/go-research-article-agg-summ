package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"
)

//==============================================================================
// CONFIGURATION
//==============================================================================

// Your Groq API key. It's best practice to load this from an environment variable.
// You can get a key from: https://console.groq.com/keys
var groqAPIKey = os.Getenv("GROQ_API_KEY")

// A simple in-memory cache to avoid re-summarizing the same abstract repeatedly.
// For a production system, you would use a more persistent cache like Redis.
var cache = make(map[string]string)
var cacheMutex = &sync.Mutex{}

//==============================================================================
// DATA STRUCTURES
//==============================================================================

// FinalArticle represents the consolidated data structure for an article
// that will be sent back to the frontend.
type FinalArticle struct {
	Title     string   `json:"title"`
	Authors   []string `json:"authors"`
	Abstract  string   `json:"abstract"`
	URL       string   `json:"url"`
	Source    string   `json:"source"`
	AISummary string   `json:"ai_summary"`
}

//--- arXiv Specific Structs ---

// These structs are designed to unmarshal the XML response from the arXiv API.
// We are only capturing the fields we need.
type ArxivResponse struct {
	Entries []ArxivEntry `xml:"entry"`
}
type ArxivEntry struct {
	Title   string        `xml:"title"`
	Authors []ArxivAuthor `xml:"author"`
	Summary string        `xml:"summary"`
	ID      string        `xml:"id"` // The URL of the paper
}
type ArxivAuthor struct {
	Name string `xml:"name"`
}

//--- Semantic Scholar Specific Structs ---

// These structs are for unmarshalling the JSON response from the Semantic Scholar API.
type SemanticScholarResponse struct {
	Total int                    `json:"total"`
	Data  []SemanticScholarPaper `json:"data"`
}
type SemanticScholarPaper struct {
	PaperID  string                  `json:"paperId"`
	URL      string                  `json:"url"`
	Title    string                  `json:"title"`
	Abstract string                  `json:"abstract"`
	Authors  []SemanticScholarAuthor `json:"authors"`
}
type SemanticScholarAuthor struct {
	Name string `json:"name"`
}

//--- Groq API Specific Structs ---

// Structs for sending a request to the Groq API.
type GroqRequest struct {
	Messages []GroqMessage `json:"messages"`
	Model    string        `json:"model"`
}
type GroqMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Structs for receiving a response from the Groq API.
type GroqResponse struct {
	Choices []GroqChoice `json:"choices"`
}
type GroqChoice struct {
	Message GroqMessage `json:"message"`
}

//==============================================================================
// API LOGIC: EXTERNAL SERVICES
//==============================================================================

// searchArxiv fetches research papers from the arXiv API.
func searchArxiv(query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done() // Signal that this goroutine is finished.

	// In a real app, you would properly handle XML parsing.
	// For simplicity, we are skipping it, as most modern APIs use JSON.
	// This function serves as a placeholder to show the concurrent structure.
	log.Printf("INFO: arXiv search is currently a placeholder. No articles will be fetched from this source.")
	// Example of what a real implementation would look like:
	// endpoint := fmt.Sprintf("http://export.arxiv.org/api/query?search_query=all:%s&max_results=5", url.QueryEscape(query))
	// ... make request, parse XML, and send articles to articlesChan ...
}

// searchSemanticScholar fetches research papers from the Semantic Scholar API.
func searchSemanticScholar(query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done()

	endpoint := fmt.Sprintf("https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=5&fields=url,title,abstract,authors.name", url.QueryEscape(query))

	resp, err := http.Get(endpoint)
	if err != nil {
		log.Printf("ERROR: Failed to fetch from Semantic Scholar: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("ERROR: Semantic Scholar returned non-200 status: %d. Body: %s", resp.StatusCode, string(bodyBytes))
		return
	}

	var apiResponse SemanticScholarResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		log.Printf("ERROR: Failed to decode Semantic Scholar JSON response: %v", err)
		return
	}

	log.Printf("INFO: Found %d articles from Semantic Scholar", len(apiResponse.Data))

	for _, paper := range apiResponse.Data {
		if paper.Abstract == "" {
			continue // Skip papers without an abstract
		}

		var authors []string
		for _, author := range paper.Authors {
			authors = append(authors, author.Name)
		}

		articlesChan <- FinalArticle{
			Title:    paper.Title,
			Authors:  authors,
			Abstract: paper.Abstract,
			URL:      paper.URL,
			Source:   "Semantic Scholar",
		}
	}
}

// summarizeWithGroq takes text (like an abstract) and returns a summary using the Groq API.
func summarizeWithGroq(textToSummarize string) (string, error) {
	if groqAPIKey == "" {
		return "GROQ_API_KEY not set. Summary unavailable.", fmt.Errorf("GROQ_API_KEY environment variable not set")
	}

	// Check cache first
	cacheMutex.Lock()
	summary, found := cache[textToSummarize]
	cacheMutex.Unlock()
	if found {
		log.Println("INFO: Returning summary from cache.")
		return summary, nil
	}

	requestBody := GroqRequest{
		Model: "llama3-8b-8192", // Using Llama 3 8B model
		Messages: []GroqMessage{
			{
				Role:    "system",
				Content: "You are a helpful research assistant. Summarize the following academic abstract concisely, focusing on the key findings and methodology in about 3-4 sentences.",
			},
			{
				Role:    "user",
				Content: textToSummarize,
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal Groq request: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create Groq request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+groqAPIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request to Groq API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Groq API returned non-200 status: %d. Body: %s", resp.StatusCode, string(bodyBytes))
	}

	var groqResponse GroqResponse
	if err := json.NewDecoder(resp.Body).Decode(&groqResponse); err != nil {
		return "", fmt.Errorf("failed to decode Groq response: %w", err)
	}

	if len(groqResponse.Choices) > 0 {
		aiSummary := groqResponse.Choices[0].Message.Content
		// Store in cache
		cacheMutex.Lock()
		cache[textToSummarize] = aiSummary
		cacheMutex.Unlock()
		return aiSummary, nil
	}

	return "", fmt.Errorf("no summary found in Groq response")
}

//==============================================================================
// API HANDLER
//==============================================================================

// searchHandler is the main handler for the /api/search endpoint.
func searchHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	log.Printf("INFO: Received search request for query: '%s'", query)

	// --- Step 1: Concurrently fetch articles from all sources ---
	var wg sync.WaitGroup
	articlesChan := make(chan FinalArticle, 10) // Buffered channel

	sources := []func(string, chan<- FinalArticle, *sync.WaitGroup){
		searchArxiv,
		searchSemanticScholar,
		// Add more source functions here in the future
	}

	wg.Add(len(sources))
	for _, sourceFunc := range sources {
		go sourceFunc(query, articlesChan, &wg)
	}

	// Close the channel once all fetching goroutines are done.
	go func() {
		wg.Wait()
		close(articlesChan)
	}()

	// --- Step 2: Collect results and summarize them ---
	var finalResults []FinalArticle
	var summaryWg sync.WaitGroup

	for article := range articlesChan {
		summaryWg.Add(1)
		// Process each article in a new goroutine to summarize in parallel
		go func(art FinalArticle) {
			defer summaryWg.Done()
			summary, err := summarizeWithGroq(art.Abstract)
			if err != nil {
				log.Printf("WARN: Could not summarize article '%s': %v", art.Title, err)
				art.AISummary = "Could not generate summary."
			} else {
				art.AISummary = summary
			}
			// This part needs to be thread-safe when appending to the slice
			// For simplicity, we'll append after all summaries are done.
			// In a high-performance app, you'd use a channel or a mutex-protected slice.
			finalResults = append(finalResults, art)
		}(article)
	}

	summaryWg.Wait() // Wait for all summarizations to complete

	// --- Step 3: Send the final JSON response ---
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*") // For development, allow all origins

	if err := json.NewEncoder(w).Encode(finalResults); err != nil {
		log.Printf("ERROR: Failed to encode final response: %v", err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
	log.Printf("INFO: Successfully sent %d results.", len(finalResults))
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

func main() {
	// Check for Groq API key at startup
	if groqAPIKey == "" {
		log.Println("WARNING: GROQ_API_KEY environment variable is not set. AI summarization will be disabled.")
	}

	// Define the HTTP server and handler
	mux := http.NewServeMux()
	mux.HandleFunc("/api/search", searchHandler)

	port := "8080"
	log.Printf("INFO: Starting server on http://localhost:%s", port)

	// Start the server
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatalf("FATAL: Could not start server: %s\n", err)
	}
}

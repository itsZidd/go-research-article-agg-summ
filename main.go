package main

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
)

//==============================================================================
// CONFIGURATION
//==============================================================================

var groqAPIKey = os.Getenv("GROQ_API_KEY")

var cache = make(map[string]string)
var cacheMutex = &sync.Mutex{}

var httpClient = &http.Client{
	Timeout: 20 * time.Second,
}

// A simple regex to remove non-alphanumeric characters for deduplication keys.
var nonAlphanumericRegex = regexp.MustCompile(`[^a-zA-Z0-9]+`)

//==============================================================================
// DATA STRUCTURES
//==============================================================================

type APIResponse struct {
	TotalResults     int            `json:"total_results"`
	SearchTimeMillis int64          `json:"search_time_ms"`
	Articles         []FinalArticle `json:"articles"`
}

type FinalArticle struct {
	Title     string   `json:"title"`
	Authors   []string `json:"authors"`
	Abstract  string   `json:"abstract"`
	URL       string   `json:"url"`
	Source    string   `json:"source"`
	AISummary string   `json:"ai_summary"`
}

// --- arXiv Specific Structs ---
type ArxivResponse struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []ArxivEntry `xml:"entry"`
}
type ArxivEntry struct {
	Title   string        `xml:"title"`
	Authors []ArxivAuthor `xml:"author"`
	Summary string        `xml:"summary"`
	ID      string        `xml:"id"`
}
type ArxivAuthor struct {
	Name string `xml:"name"`
}

// --- Semantic Scholar Specific Structs ---
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

// --- PLOS Specific Structs ---
type PlosResponse struct {
	Response struct {
		Docs []PlosDoc `json:"docs"`
	} `json:"response"`
}
type PlosDoc struct {
	ID       string   `json:"id"` // This is the DOI
	Title    string   `json:"title_display"`
	Authors  []string `json:"author_display"`
	Abstract []string `json:"abstract"`
}

// --- DOAJ Specific Structs ---
type DoajResponse struct {
	Results []DoajArticle `json:"results"`
}
type DoajArticle struct {
	BibJSON struct {
		Title  string `json:"title"`
		Author []struct {
			Name string `json:"name"`
		} `json:"author"`
		Abstract string `json:"abstract"`
		Link     []struct {
			Type string `json:"type"`
			URL  string `json:"url"`
		} `json:"link"`
	} `json:"bibjson"`
}

// --- Groq API Specific Structs ---
type GroqRequest struct {
	Messages []GroqMessage `json:"messages"`
	Model    string        `json:"model"`
}
type GroqMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type GroqResponse struct {
	Choices []GroqChoice `json:"choices"`
}
type GroqChoice struct {
	Message GroqMessage `json:"message"`
}

//==============================================================================
// API LOGIC: EXTERNAL SERVICES
//==============================================================================

// All search functions now accept a `context.Context` to handle cancellations.
func searchArxiv(ctx context.Context, query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done()
	endpoint := fmt.Sprintf("http://export.arxiv.org/api/query?search_query=all:%s&start=0&max_results=5", url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		log.Printf("ERROR: Failed to create arXiv request: %v", err)
		return
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("ERROR: Failed to fetch from arXiv: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("ERROR: arXiv returned non-200 status: %d", resp.StatusCode)
		return
	}
	var apiResponse ArxivResponse
	if err := xml.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		log.Printf("ERROR: Failed to decode arXiv XML: %v", err)
		return
	}
	log.Printf("INFO: Found %d articles from arXiv", len(apiResponse.Entries))
	for _, entry := range apiResponse.Entries {
		cleanAbstract := strings.TrimSpace(strings.ReplaceAll(entry.Summary, "\n", " "))
		if cleanAbstract == "" {
			continue
		}
		var authors []string
		for _, author := range entry.Authors {
			authors = append(authors, author.Name)
		}
		articlesChan <- FinalArticle{
			Title:    strings.TrimSpace(entry.Title),
			Authors:  authors,
			Abstract: cleanAbstract,
			URL:      entry.ID,
			Source:   "arXiv",
		}
	}
}

func searchSemanticScholar(ctx context.Context, query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done()
	endpoint := fmt.Sprintf("https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=5&fields=url,title,abstract,authors.name", url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		log.Printf("ERROR: Failed to create Semantic Scholar request: %v", err)
		return
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("ERROR: Failed to fetch from Semantic Scholar: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("ERROR: Semantic Scholar returned non-200 status: %d", resp.StatusCode)
		return
	}
	var apiResponse SemanticScholarResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		log.Printf("ERROR: Failed to decode Semantic Scholar JSON: %v", err)
		return
	}
	log.Printf("INFO: Found %d articles from Semantic Scholar", len(apiResponse.Data))
	for _, paper := range apiResponse.Data {
		if paper.Abstract == "" {
			continue
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

func searchPlos(ctx context.Context, query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done()
	endpoint := fmt.Sprintf("https://api.plos.org/search?q=everything:%s&rows=5&fl=id,title_display,author_display,abstract", url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		log.Printf("ERROR: Failed to create PLOS request: %v", err)
		return
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("ERROR: Failed to fetch from PLOS: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("ERROR: PLOS returned non-200 status: %d", resp.StatusCode)
		return
	}
	var apiResponse PlosResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		log.Printf("ERROR: Failed to decode PLOS JSON: %v", err)
		return
	}
	log.Printf("INFO: Found %d articles from PLOS", len(apiResponse.Response.Docs))
	for _, doc := range apiResponse.Response.Docs {
		if len(doc.Abstract) == 0 || doc.Abstract[0] == "" {
			continue
		}
		articlesChan <- FinalArticle{
			Title:    doc.Title,
			Authors:  doc.Authors,
			Abstract: doc.Abstract[0],
			URL:      "https://doi.org/" + doc.ID,
			Source:   "PLOS",
		}
	}
}

func searchDoaj(ctx context.Context, query string, articlesChan chan<- FinalArticle, wg *sync.WaitGroup) {
	defer wg.Done()
	endpoint := fmt.Sprintf("https://doaj.org/api/search/articles/%s?pageSize=5", url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		log.Printf("ERROR: Failed to create DOAJ request: %v", err)
		return
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("ERROR: Failed to fetch from DOAJ: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("ERROR: DOAJ returned non-200 status: %d", resp.StatusCode)
		return
	}
	var apiResponse DoajResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		log.Printf("ERROR: Failed to decode DOAJ JSON: %v", err)
		return
	}
	log.Printf("INFO: Found %d articles from DOAJ", len(apiResponse.Results))
	for _, article := range apiResponse.Results {
		bibjson := article.BibJSON
		if bibjson.Abstract == "" {
			continue
		}
		var authors []string
		for _, author := range bibjson.Author {
			authors = append(authors, author.Name)
		}
		var fulltextURL string
		for _, link := range bibjson.Link {
			if link.Type == "fulltext" {
				fulltextURL = link.URL
				break
			}
		}
		if fulltextURL == "" && len(bibjson.Link) > 0 {
			fulltextURL = bibjson.Link[0].URL
		}
		articlesChan <- FinalArticle{
			Title:    bibjson.Title,
			Authors:  authors,
			Abstract: bibjson.Abstract,
			URL:      fulltextURL,
			Source:   "DOAJ",
		}
	}
}

func summarizeWithGroq(ctx context.Context, textToSummarize string) (string, error) {
	if groqAPIKey == "" {
		return "GROQ_API_KEY not set. Summary unavailable.", fmt.Errorf("GROQ_API_KEY environment variable not set")
	}
	cacheMutex.Lock()
	summary, found := cache[textToSummarize]
	cacheMutex.Unlock()
	if found {
		log.Println("INFO: Returning summary from cache.")
		return summary, nil
	}
	requestBody := GroqRequest{
		Model: "llama3-8b-8192",
		Messages: []GroqMessage{
			{Role: "system", Content: "You are a helpful research assistant. Summarize the following academic abstract concisely, focusing on the key findings and methodology in about 3-4 sentences."},
			{Role: "user", Content: textToSummarize},
		},
	}
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal Groq request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.groq.com/openai/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create Groq request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+groqAPIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
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

func searchHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context() // This is the request context. It's used for cancellation.

	query := r.URL.Query().Get("q")
	if len(query) < 3 || len(query) > 200 {
		http.Error(w, "Query parameter 'q' must be between 3 and 200 characters", http.StatusBadRequest)
		return
	}
	log.Printf("INFO: Received search request for query: '%s'", query)

	var fetchWg sync.WaitGroup
	rawArticlesChan := make(chan FinalArticle, 25)

	sources := []func(context.Context, string, chan<- FinalArticle, *sync.WaitGroup){
		searchArxiv,
		searchSemanticScholar,
		searchPlos,
		searchDoaj,
	}

	fetchWg.Add(len(sources))
	for _, sourceFunc := range sources {
		go sourceFunc(ctx, query, rawArticlesChan, &fetchWg)
	}

	go func() {
		fetchWg.Wait()
		close(rawArticlesChan)
	}()

	var summarizeWg sync.WaitGroup
	summarizedArticlesChan := make(chan FinalArticle, 25)

	for article := range rawArticlesChan {
		summarizeWg.Add(1)
		go func(art FinalArticle) {
			defer summarizeWg.Done()
			summary, err := summarizeWithGroq(ctx, art.Abstract)
			if err != nil {
				// Check if the error was due to context cancellation
				if ctx.Err() != nil {
					log.Println("INFO: Summarization cancelled.")
					return
				}
				log.Printf("WARN: Could not summarize article '%s': %v", art.Title, err)
				art.AISummary = "Could not generate summary."
			} else {
				art.AISummary = summary
			}
			summarizedArticlesChan <- art
		}(article)
	}

	go func() {
		summarizeWg.Wait()
		close(summarizedArticlesChan)
	}()

	var collectedArticles []FinalArticle
	for article := range summarizedArticlesChan {
		collectedArticles = append(collectedArticles, article)
	}

	// --- Deduplication Logic ---
	// Use a map to track unique articles based on a normalized title.
	uniqueArticles := make(map[string]FinalArticle)
	for _, article := range collectedArticles {
		// Normalize title to use as a key: lowercase and remove non-alphanumeric chars.
		key := nonAlphanumericRegex.ReplaceAllString(strings.ToLower(article.Title), "")
		if _, exists := uniqueArticles[key]; !exists {
			uniqueArticles[key] = article
		}
	}

	// Convert map back to a slice
	var finalArticles []FinalArticle
	for _, article := range uniqueArticles {
		finalArticles = append(finalArticles, article)
	}

	duration := time.Since(start).Milliseconds()
	finalResponse := APIResponse{
		TotalResults:     len(finalArticles),
		SearchTimeMillis: duration,
		Articles:         finalArticles,
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	if err := json.NewEncoder(w).Encode(finalResponse); err != nil {
		log.Printf("ERROR: Failed to encode final response: %v", err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
	log.Printf("INFO: Successfully sent %d unique results in %dms.", len(finalArticles), duration)
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

func main() {
	if groqAPIKey == "" {
		log.Println("WARNING: GROQ_API_KEY environment variable is not set. AI summarization will be disabled.")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/search", searchHandler)
	port := "8080"
	log.Printf("INFO: Starting server on http://localhost:%s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatalf("FATAL: Could not start server: %s\n", err)
	}
}

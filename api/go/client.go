// CodeAgent-RWKV Go client
// Usage: go run client.go [prompt]
//
// Build: go build -o codeagent client.go

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const (
	API_URL  = "http://127.0.0.1:8080/v1/chat"
	API_COMP = "http://127.0.0.1:8080/v1/complete"
)

type ChatRequest struct {
	Prompt      string  `json:"prompt"`
	System      string  `json:"system,omitempty"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	Stream      bool    `json:"stream"`
}

type ChatResponse struct {
	Text  string `json:"text"`
	Usage struct {
		Tokens int     `json:"tokens"`
		Time   float64 `json:"time"`
	} `json:"usage"`
}

func chat(prompt string, stream bool) (*ChatResponse, error) {
	req := ChatRequest{
		Prompt:      prompt,
		System:      "You are a helpful coding assistant. Think step by step.",
		MaxTokens:   512,
		Temperature: 0.7,
		TopP:        0.9,
		Stream:      stream,
	}

	body, _ := json.Marshal(req)
	url := API_URL
	if stream {
		url = API_URL + "/stream"
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if stream {
		return readStream(resp.Body)
	}

	var result ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode failed: %w", err)
	}
	return &result, nil
}

func readStream(body io.Reader) (*ChatResponse, error) {
	result := &ChatResponse{}
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		json.Unmarshal([]byte(data), result)
		fmt.Print(result.Text)
	}
	return result, scanner.Err()
}

func complete(prompt string) (*ChatResponse, error) {
	req := ChatRequest{
		Prompt:      prompt,
		MaxTokens:   512,
		Temperature: 0.7,
		TopP:        0.9,
	}
	body, _ := json.Marshal(req)
	resp, err := http.Post(API_COMP, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result ChatResponse
	json.NewDecoder(resp.Body).Decode(&result)
	return &result, nil
}

func main() {
	server := os.Getenv("CODEGENT_API")
	if server != "" {
		os.Setenv("API_URL", server+"/v1/chat")
	}

	prompt := "Write a hello world in Python"
	if len(os.Args) > 1 {
		prompt = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("CodeAgent > %s\n\n", prompt)
	resp, err := chat(prompt, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(resp.Text)
	fmt.Printf("\n[%d tokens, %.1fs]\n", resp.Usage.Tokens, resp.Usage.Time)
}

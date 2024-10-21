use crate::{
    error::LlamaCoreError, graph::Graph, metadata::ggml::GgmlMetadata, utils::get_output_buffer,
    CHAT_GRAPHS,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Perform a web search with a use input
pub async fn search<S: AsRef<str> + Send, E: SearchEngine + Sync>(
    search_engine: &E,
    input: S,
) -> Result<SearchOutput, LlamaCoreError> {
    search_engine.search(input).await
}

/// Summarize the search results
pub fn summarize<E: SearchEngine + Sync>(
    search_engine: &E,
    search_output: SearchOutput,
    head_prompt: Option<&str>,
    tail_prompt: Option<&str>,
) -> Result<String, LlamaCoreError> {
    search_engine.summarize(search_output, head_prompt, tail_prompt)
}

/// Search and summarize the search results
pub async fn search_and_summarize<S: AsRef<str> + Send, E: SearchEngine + Sync>(
    search_engine: &E,
    input: S,
    head_prompt: Option<String>,
    tail_prompt: Option<String>,
) -> Result<String, LlamaCoreError> {
    search_engine
        .search_and_summarize(input, head_prompt, tail_prompt)
        .await
}

/// Possible input/output Content Types. Currently only supports JSON.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ContentType {
    JSON,
}
impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                ContentType::JSON => "application/json",
            }
        )
    }
}

#[async_trait]
pub trait SearchEngine {
    async fn search<S: AsRef<str> + Send>(&self, input: S) -> Result<SearchOutput, LlamaCoreError>;

    fn parse_raw_results(
        &self,
        raw_results: &serde_json::Value,
    ) -> Result<SearchOutput, LlamaCoreError>;

    fn summarize(
        &self,
        search_output: SearchOutput,
        head_prompt: Option<&str>,
        tail_prompt: Option<&str>,
    ) -> Result<String, LlamaCoreError> {
        let mut search_output_string: String = String::new();

        // Add the text content of every result together.
        search_output
            .results
            .iter()
            .for_each(|result| search_output_string.push_str(result.text_content.as_str()));

        // Error on embedding running mode.
        if crate::running_mode()? == crate::RunningMode::Embeddings {
            let err_msg = "Summarization is not supported in the EMBEDDINGS running mode.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Search(err_msg.into()));
        }

        // Get graphs and pick the first graph.
        let chat_graphs = match CHAT_GRAPHS.get() {
            Some(chat_graphs) => chat_graphs,
            None => {
                let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Search(err_msg.into()));
            }
        };

        let mut chat_graphs = chat_graphs.lock().map_err(|e| {
            let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Search(err_msg)
        })?;

        // Use first available chat graph
        let graph: &mut Graph<GgmlMetadata> = match chat_graphs.values_mut().next() {
            Some(graph) => graph,
            None => {
                let err_msg = "No available chat graph.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Search(err_msg.into()));
            }
        };

        let head_prompt =
            head_prompt.unwrap_or("The following are search results I found on the internet:");

        let tail_prompt = tail_prompt.unwrap_or("To sum up them up: ");

        // Prepare input prompt.
        let input = format!(
            "{}\n\n{}\n\n{}",
            head_prompt, search_output_string, tail_prompt
        );
        let tensor_data = input.trim().as_bytes().to_vec();

        graph
            .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
            .expect("Failed to set prompt as the input tensor");

        #[cfg(feature = "logging")]
        info!(target: "stdout", "Generating a summary for search results...");
        // Execute the inference.
        graph.compute().expect("Failed to complete inference");

        let output_buffer = get_output_buffer(graph, 0)?;

        // Compute lossy UTF-8 output (text only).
        let output = String::from_utf8_lossy(&output_buffer[..]).to_string();

        #[cfg(feature = "logging")]
        info!(target: "stdout", "Summary generated.");

        Ok(output)
    }

    async fn search_and_summarize<S: AsRef<str> + Send>(
        &self,
        input: S,
        head_prompt: Option<String>,
        tail_prompt: Option<String>,
    ) -> Result<String, LlamaCoreError> {
        let search_output = self.search(input).await?;

        self.summarize(
            search_output,
            head_prompt.as_deref(),
            tail_prompt.as_deref(),
        )
    }
}

// define a sub module named tavily_search
pub mod tavily_search {
    use super::{SearchEngine, SearchOutput, SearchResult};
    use crate::error::LlamaCoreError;
    use async_trait::async_trait;
    use reqwest::{Client, Method, Url};
    use serde_json::json;

    #[derive(Debug)]
    pub struct TavilySearchBuilder {
        engine: TavilySearch,
    }
    impl TavilySearchBuilder {
        pub fn new(api_key: impl Into<String>) -> Self {
            let engine = TavilySearch {
                api_key: api_key.into(),
                ..Default::default()
            };

            Self { engine }
        }

        pub fn include_answer(mut self, include_answer: bool) -> Self {
            self.engine.include_answer = include_answer;
            self
        }

        pub fn include_images(mut self, include_images: bool) -> Self {
            self.engine.include_images = include_images;
            self
        }

        pub fn include_raw_content(mut self, include_raw_content: bool) -> Self {
            self.engine.include_raw_content = include_raw_content;
            self
        }

        pub fn with_max_results(mut self, max_results: u8) -> Self {
            self.engine.max_search_results = max_results;
            self
        }

        pub fn with_search_depth(mut self, search_depth: String) -> Self {
            self.engine.search_depth = search_depth;
            self
        }

        pub fn with_size_per_result(mut self, size_per_result: u16) -> Self {
            self.engine.size_per_result = size_per_result;
            self
        }

        pub fn with_endpoint(mut self, endpoint: String) -> Self {
            self.engine.endpoint = endpoint;
            self
        }

        pub fn with_additional_headers(
            mut self,
            additional_headers: std::collections::HashMap<String, String>,
        ) -> Self {
            self.engine.additional_headers = Some(additional_headers);
            self
        }

        pub fn build(self) -> TavilySearch {
            self.engine
        }
    }

    #[derive(Debug)]
    pub struct TavilySearch {
        api_key: String,
        include_answer: bool,
        include_images: bool,
        include_raw_content: bool,
        /// Maximum number search results to use. Defaults to 5.
        max_search_results: u8,
        /// The search depth to use. Defaults to "advanced".
        search_depth: String,
        /// Size to clip every result to. Defaults to 300.
        size_per_result: u16,
        /// The endpoint for the search API.
        endpoint: String,
        /// Additional headers for any other purpose.
        additional_headers: Option<std::collections::HashMap<String, String>>,
    }
    impl std::default::Default for TavilySearch {
        fn default() -> Self {
            Self {
                api_key: "".to_string(),
                include_answer: false,
                include_images: false,
                include_raw_content: false,
                max_search_results: 5,
                search_depth: "advanced".to_string(),
                size_per_result: 300,
                endpoint: "https://api.tavily.com/search".to_string(),
                additional_headers: None,
            }
        }
    }
    #[async_trait]
    impl SearchEngine for TavilySearch {
        async fn search<S: AsRef<str> + Send>(
            &self,
            input: S,
        ) -> Result<SearchOutput, LlamaCoreError> {
            let client = Client::new();
            let url = match Url::parse(&self.endpoint) {
                Ok(url) => url,
                Err(_) => {
                    let msg = "Malformed endpoint url";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When parsing endpoint url: {}",
                        msg
                    )));
                }
            };

            // create a HTTP POST request
            let mut req = client.request(Method::POST, url);

            // check headers.
            req = req.headers(
                match (&self.additional_headers.clone().unwrap_or_default()).try_into() {
                    Ok(headers) => headers,
                    Err(_) => {
                        let msg = "Failed to convert headers from HashMaps to HeaderMaps";
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "On converting headers: {}",
                            msg
                        )));
                    }
                },
            );

            // check if api_key is empty.
            if self.api_key.is_empty() {
                let msg = "Fail to perform search. API key is empty.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", msg);

                return Err(LlamaCoreError::Search(msg.into()));
            }

            // create search input
            let search_input = json!({
                "api_key": self.api_key,
                "include_answer": self.include_answer,
                "include_images": self.include_images,
                "query": input.as_ref(),
                "max_results": self.max_search_results,
                "include_raw_content": self.include_raw_content,
                "search_depth": self.search_depth,
            });

            // For POST requests, search_input goes into the request body.
            let req = req.json(&search_input);

            let response = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When recieving response: {}",
                        msg
                    )));
                }
            };

            match response.content_length() {
                Some(length) => {
                    if length == 0 {
                        let msg = "Empty response from server";
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "Unexpected content length: {}",
                            msg
                        )));
                    }
                }
                None => {
                    let msg = "Content length returned None";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "Content length field not found: {}",
                        msg
                    )));
                }
            }

            let body_text = match response.text().await {
                Ok(body) => body,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When accessing response body: {}",
                        msg
                    )));
                }
            };
            println!("{}", body_text);
            let raw_results: serde_json::Value = match serde_json::from_str(body_text.as_str()) {
                Ok(value) => value,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When converting to a JSON object: {}",
                        msg
                    )));
                }
            };

            // start cleaning the output.

            // produce SearchOutput instance with the raw results obtained from the endpoint.
            let mut search_output: SearchOutput = match self.parse_raw_results(&raw_results) {
                Ok(search_output) => search_output,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When calling parse_into_results: {}",
                        msg
                    )));
                }
            };

            // apply maximum search result limit.
            search_output
                .results
                .truncate(self.max_search_results as usize);

            // apply per result character limit.
            //
            // since the clipping only happens when split_at_checked() returns Some, the results will
            // remain unchanged should split_at_checked() return None.
            for search_result in search_output.results.iter_mut() {
                if let Some(clipped_content) = search_result
                    .text_content
                    .split_at_checked(self.size_per_result as usize)
                {
                    search_result.text_content = clipped_content.0.to_string();
                }
            }

            // Search Output cleaned and finalized.
            Ok(search_output)
        }

        fn parse_raw_results(
            &self,
            raw_results: &serde_json::Value,
        ) -> Result<SearchOutput, LlamaCoreError> {
            let results_array = match raw_results["results"].as_array() {
                Some(array) => array,
                None => {
                    let msg = "No results returned from server";
                    error!(target: "search_server", "google_parser: {}", msg);
                    return Err(LlamaCoreError::Search(msg.to_string()));
                }
            };

            let mut results = Vec::new();

            for result in results_array {
                let current_result = SearchResult {
                    url: result["url"].to_string(),
                    site_name: result["title"].to_string(),
                    text_content: result["content"].to_string(),
                };
                results.push(current_result)
            }

            Ok(SearchOutput { results })
        }
    }
}

pub mod bing_search {
    use super::{SearchEngine, SearchOutput, SearchResult};
    use crate::error::LlamaCoreError;
    use async_trait::async_trait;
    use reqwest::{Client, Method, Url};
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Debug)]
    pub struct BingSearchBuilder {
        engine: BingSearch,
    }
    impl BingSearchBuilder {
        pub fn new(api_key: impl Into<String>) -> Self {
            let mut additional_headers = HashMap::new();
            additional_headers.insert("Ocp-Apim-Subscription-Key".to_string(), api_key.into());

            let engine = BingSearch {
                additional_headers: Some(additional_headers),
                ..Default::default()
            };

            Self { engine }
        }

        pub fn with_max_results(mut self, max_results: u8) -> Self {
            self.engine.max_search_results = max_results;
            self
        }

        pub fn with_size_per_result(mut self, size_per_result: u16) -> Self {
            self.engine.size_per_result = size_per_result;
            self
        }

        pub fn with_endpoint(mut self, endpoint: String) -> Self {
            self.engine.endpoint = endpoint;
            self
        }

        pub fn with_additional_headers(
            mut self,
            additional_headers: HashMap<String, String>,
        ) -> Self {
            if self.engine.additional_headers.is_none() {
                self.engine.additional_headers = Some(additional_headers);
            } else {
                let mut current_headers = self.engine.additional_headers.unwrap();
                current_headers.extend(additional_headers);
                self.engine.additional_headers = Some(current_headers);
            }
            self
        }

        pub fn build(self) -> BingSearch {
            self.engine
        }
    }

    #[derive(Debug)]
    pub struct BingSearch {
        /// Maximum number search results to use. Defaults to 5.
        max_search_results: u8,
        /// Size to clip every result to. Defaults to 300.
        size_per_result: u16,
        /// The endpoint for the search API.
        endpoint: String,
        /// Additional headers for any other purpose.
        additional_headers: Option<std::collections::HashMap<String, String>>,
    }
    impl std::default::Default for BingSearch {
        fn default() -> Self {
            Self {
                max_search_results: 5,
                size_per_result: 300,
                endpoint: "https://api.bing.microsoft.com/v7.0/search".to_string(),
                additional_headers: None,
            }
        }
    }
    #[async_trait]
    impl SearchEngine for BingSearch {
        async fn search<S: AsRef<str> + Send>(
            &self,
            input: S,
        ) -> Result<SearchOutput, LlamaCoreError> {
            let client = Client::new();
            let url = match Url::parse(&self.endpoint) {
                Ok(url) => url,
                Err(_) => {
                    let msg = "Malformed endpoint url";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When parsing endpoint url: {}",
                        msg
                    )));
                }
            };

            let mut req = client.request(Method::POST, url);

            // check if `Ocp-Apim-Subscription-Key` is in the additional headers.
            let msg = "Fail to perform search. Not found 'Ocp-Apim-Subscription-Key' in additional headers.";
            if self.additional_headers.as_ref().map_or(true, |headers| {
                !headers.contains_key("Ocp-Apim-Subscription-Key")
            }) {
                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", msg);

                return Err(LlamaCoreError::Search(msg.into()));
            }

            // check headers.
            req = req.headers(
                match (&self.additional_headers.clone().unwrap()).try_into() {
                    Ok(headers) => headers,
                    Err(_) => {
                        let msg = "Failed to convert headers from HashMaps to HeaderMaps";
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "On converting headers: {}",
                            msg
                        )));
                    }
                },
            );

            // create search input
            let search_input = json!({
                "count": self.max_search_results,
                "q": input.as_ref(),
                "responseFilter": "Webpages",
            });

            // For POST requests, search_input goes into the request body.
            let req = req.json(&search_input);

            let response = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When recieving response: {}",
                        msg
                    )));
                }
            };

            match response.content_length() {
                Some(length) => {
                    if length == 0 {
                        let msg = "Empty response from server";
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "Unexpected content length: {}",
                            msg
                        )));
                    }
                }
                None => {
                    let msg = "Content length returned None";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "Content length field not found: {}",
                        msg
                    )));
                }
            }

            let body_text = match response.text().await {
                Ok(body) => body,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When accessing response body: {}",
                        msg
                    )));
                }
            };
            println!("{}", body_text);
            let raw_results: serde_json::Value = match serde_json::from_str(body_text.as_str()) {
                Ok(value) => value,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When converting to a JSON object: {}",
                        msg
                    )));
                }
            };

            // start cleaning the output.

            // produce SearchOutput instance with the raw results obtained from the endpoint.
            let mut search_output: SearchOutput = match self.parse_raw_results(&raw_results) {
                Ok(search_output) => search_output,
                Err(e) => {
                    let msg = e.to_string();
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "When calling parse_into_results: {}",
                        msg
                    )));
                }
            };

            // apply maximum search result limit.
            search_output
                .results
                .truncate(self.max_search_results as usize);

            // apply per result character limit.
            //
            // since the clipping only happens when split_at_checked() returns Some, the results will
            // remain unchanged should split_at_checked() return None.
            for result in search_output.results.iter_mut() {
                if let Some(clipped_content) = result
                    .text_content
                    .split_at_checked(self.size_per_result as usize)
                {
                    result.text_content = clipped_content.0.to_string();
                }
            }

            // Search Output cleaned and finalized.
            Ok(search_output)
        }

        fn parse_raw_results(
            &self,
            raw_results: &serde_json::Value,
        ) -> Result<SearchOutput, LlamaCoreError> {
            // parse webpages
            let web_pages_object = match raw_results["webPages"].is_object() {
                true => match raw_results["webPages"]["value"].as_array() {
                    Some(value) => value,
                    None => {
                        let msg =
                            r#"could not convert the "value" field of "webPages" to an array"#;
                        error!(target: "bing_parser", "bing_parser: {}", msg);
                        return Err(LlamaCoreError::Operation(msg.to_string()));
                    }
                },
                false => {
                    let msg = "no webpages found when parsing query.";
                    error!(target: "bing_parser", "bing_parser: {}", msg);
                    return Err(LlamaCoreError::Operation(msg.to_string()));
                }
            };

            let mut results = Vec::new();
            for result in web_pages_object {
                let current_result = SearchResult {
                    url: result["url"].to_string(),
                    site_name: result["siteName"].to_string(),
                    text_content: result["snippet"].to_string(),
                };
                results.push(current_result);
            }

            Ok(SearchOutput { results })
        }
    }
}

/// output format for individual results in the final output.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub url: String,
    pub site_name: String,
    pub text_content: String,
}

/// Final output format for consumption by the LLM.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
}

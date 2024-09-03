use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionUserMessage,
    ChatCompletionUserMessageContent, ContentPart,
};
use regex::Regex;

/// Generate chat prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct CohereChatPrompt;
impl CohereChatPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &mut ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "{system_prompt}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_message}<|END_OF_TURN_TOKEN|>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_message}<|END_OF_TURN_TOKEN|>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls().is_some() {
                true => String::new(),
                false => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{assistant_message}<|END_OF_TURN_TOKEN|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for CohereChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let preamble = r###"# Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."###;
        let system_prompt = format!(
            "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{preamble}<|END_OF_TURN_TOKEN|>"
        );

        let len = messages.len();
        let mut context = String::new();
        match &messages.last() {
            Some(ChatCompletionRequestMessage::User(message)) => {
                if let ChatCompletionUserMessageContent::Text(content) = message.content() {
                    let re = Regex::new(r"(?s)(<result>.*?</result>)")
                        .map_err(|e| PromptError::Operation(e.to_string()))?;

                    let real_user_query = match re.find(content) {
                        Some(found) => {
                            let start = found.start();
                            // let end = found.end();
                            context = content[start..].to_string();
                            content[..start].to_string()
                        }
                        None => {
                            return Err(PromptError::Operation("No match found".to_string()));
                        }
                    };

                    // compose new user message content
                    let content = ChatCompletionUserMessageContent::Text(real_user_query);

                    // create user message
                    let user_message = ChatCompletionRequestMessage::new_user_message(
                        content,
                        message.name().cloned(),
                    );
                    // replace the original user message
                    messages[len - 1] = user_message;
                }
            }
            _ => {
                let err_msg = "The last message in the chat request is not user message.";

                return Err(PromptError::BadMessages(err_msg.to_string()));
            }
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        // append retrieved context
        prompt = format!(
            "{}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
            prompt, context
        );

        // append instructions
        let instructions = r###"Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0."###;
        prompt = format!(
            "{}|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
            prompt, instructions
        );

        prompt.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");

        Ok(prompt)
    }
}

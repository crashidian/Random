import os
import openai
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext
import threading

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Chat Application")
        self.root.geometry("800x600")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('put your API key in here please'))
        
        # Chat history
        self.messages = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main chat display
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=30)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Create input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # Create input text field
        self.input_field = tk.Entry(input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create send button
        send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Bind Enter key to send message
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
    def send_message(self):
        user_message = self.input_field.get().strip()
        if not user_message:
            return
            
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.chat_display.insert(tk.END, f"You: {user_message}\n\n")
        
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Start API call in separate thread to prevent UI freezing
        threading.Thread(target=self.get_ai_response).start()
        
    def get_ai_response(self):
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Get AI response
            ai_message = response.choices[0].message.content
            
            # Add AI response to history
            self.messages.append({"role": "assistant", "content": ai_message})
            
            # Display AI response
            self.chat_display.insert(tk.END, f"Assistant: {ai_message}\n\n")
            
            # Scroll to bottom
            self.chat_display.see(tk.END)
            
        except Exception as e:
            error_message = f"Error: {str(e)}\n\n"
            self.chat_display.insert(tk.END, error_message)
            self.chat_display.see(tk.END)

def main():
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('OPENAI_API_KEY=your_api_key_here')
            
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

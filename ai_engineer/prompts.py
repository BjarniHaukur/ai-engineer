
roadmap_prompt = '''
You will get instructions for code to write.
You will write a very long answer. Make sure that every detail of the architecture is, in the end, implemented as code.

'''

file_format_prompt = '''
You will output the content of each file necessary to achieve the goal, including ALL code.
Represent files like so:

FILENAME
```
CODE
```

The following tokens must be replaced like so:
FILENAME is the lowercase combined path and file name including the file extension
CODE is the code in the file

Example representation of a file:


```src/hello_world.py
print("Hello World")
```

Do not comment on what every file does. Please note that the code should be fully functional. No placeholders.

'''

philosophy_prompt = '''

Almost always put different classes in different files.
Always use the programming language the user asks for.
For Python, you always create an appropriate requirements.txt file.
Always add a comment briefly describing the purpose of the function definition.
Add comments explaining very complex bits of logic.
Always follow the best practices for the requested languages for folder/file structure and how to package the project.


Python toolbelt preferences:
- pytest
- dataclasses

'''

sandbox_prompt = '''

The entry point for your code is the "main.py" file.
Don't include any other text or example usages in your response.

'''

fence = ''' ########################################################################## '''
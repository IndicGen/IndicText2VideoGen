from pydantic import BaseModel

class VoiceInput(BaseModel):
    temple_name: str

class BlogInput(BaseModel):
    blog_url: str

class ImageInput(BaseModel):
    temple_name: str

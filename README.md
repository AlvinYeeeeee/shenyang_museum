# 沈阳博物馆

## Configuration
Edit the `src/conf_http_server.yaml` to set the model path, text file path, et.al.
Please see the yaml file for detailed explaination.

## How to run

1. Start ollama
```shell
ollama serve
```

2. start the web server

> `cd` to the root path of this project and run through fastapi:
```shell
cd path/to/rag-it
fastapi run --port 1888 src/fastapi_web_server.py 
```

## Response format

1. send the http request for question:

format: 
```http://116.136.130.168:31041/?input_text=question```,
replace `question` to your question, e.g. 
`http://116.136.130.168:31041/?input_text="请介绍一下沈阳"`

2. Get text response:

The response will be a json with format:
```json
{
    "response": response,
    "audio_file": response_audio_file
}
```
with the `response` being the returned answer,
and `audio_file` being the returned file path to the audio file.
E.g.:
```json
{"response":"而在这一时期最具代表性的是郑家洼子青铜短剑文化，...，可见其地位之高和生活之奢靡。",
"audio_file":"/root/epfs/rag-it/response_audios/tmp454iqfx9.wav"}
```
3. Send the http request for audio file:
`http://116.136.130.168:31041/audios/?audio_file_path=response_audio_file`
replace the `response_audio_file` with the `audio_file` value in the json reponse returned in step 2.
e.g.:
`http://116.136.130.168:31041/audios/?audio_file_path=/root/epfs/rag-it/response_audios/tmp454iqfx9.wav`

4. In WeChat miniprogram:
```html
<audio src="http://116.136.130.168:31041/audios/?audio_file_path=/root/epfs/rag-it/response_audios/tmp454iqfx9.wav" id="myAudio" controls loop></audio>
```
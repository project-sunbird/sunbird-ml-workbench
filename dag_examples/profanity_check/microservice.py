def profanity_check(input_json):
    """
    Function that takes text from request and checkes it for profanity.
    :param: input_json(json): Example input: {
                                "id": "api.ml.profanity",
                                "ver": "1.0",
                                "ts": "2019-06-30T12:40:40+05:30",
                                "params": {},
                                "request":{
                                    "text":"****"
                                    }
                                }
    :returns: output_json(json): Example output: {"identifier":"do_312593171904544768222861",
     "score": 0.0,
     "profanity": [{"method": "better_profanity", "timeOfoccurance": "NA", "sentOfoccurance": "NA", "detectedWord": "NA", "score": 0.0}, {"method": "ProfanityFilter", "timeOfoccurance": "NA", "sentOfoccurance": "NA", "detectedWord": "NA", "score": 0.0}, {"method": "profanity_check", "timeOfoccurance": "NA", "sentOfoccurance": "", "detectedWord": "NA", "score": 0.0}]}
    """
    import time
    errmsg= ""
    status = ""
    try:
        text = input_json["request"]["text"]
    except:
        status = 400
        errmsg = "InvalidRequest"

    if status !=400:
        try:
            from daggit.contrib.sunbird.oplib.profanityUtils import text_profanity
            check_profanity = text_profanity(text)
            status = 200
        except:
            status = 500
            errmsg = "Error in function call text_profanity()"

    if status ==200:
            param_status="success"
    else:
            param_status="fail"

    time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
    api_response= {
        "id": "api.org.ml",
        "ver": "v1",
        "ts": time_format,
        "params": {
            "resmsgid": "null",
            "msgid": "",
            "err": "null",
            "status": param_status,
            "errmsg": errmsg
        },
        "responseCode": "OK",
        "result": check_profanity
    }
    return api_response


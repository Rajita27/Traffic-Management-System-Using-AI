class RequestLoggerMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print(f"⚠️ Request method: {request.method}, Path: {request.path}")
        return self.get_response(request)

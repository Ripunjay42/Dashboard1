from app import create_app

app = create_app()

if __name__ == '__main__':
    # Use threaded=True for better performance with multiple connections
    # use_reloader=False prevents camera reinitialization on code changes
    # For production, use gunicorn with: gunicorn -w 4 -k gevent run:app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

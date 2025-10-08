# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# from api.routes import router
# from database.mongodb import MongoDB
# from config import settings


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     await MongoDB.connect_db()
#     yield
#     # Shutdown
#     await MongoDB.close_db()


# app = FastAPI(
#     title=settings.APP_NAME,
#     description="Prospecting Agent - Company Research API",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include routes
# app.include_router(router, prefix="/api", tags=["prospecting"])


# @app.get("/")
# async def root():
#     return {
#         "message": "Prospecting Agent API",
#         "version": "1.0.0",
#         "docs": "/docs"
#     }


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=settings.DEBUG
#     )

class MemoryServer:
    def __init__(self):
        """Initialize the server."""
        self.server = Server(SERVER_NAME)
        
        try:
            # Initialize paths
            logger.info(f"Creating directories if they don't exist...")
            os.makedirs(CHROMA_PATH, exist_ok=True)
            os.makedirs(BACKUPS_PATH, exist_ok=True)
            
            # Initialize storage - this part is synchronous
            self.storage = ChromaMemoryStorage(CHROMA_PATH)
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
        
        # Register handlers
        self.register_handlers()
        logger.info("Server initialization complete")

    async def initialize(self):
        """Async initialization method."""
        try:
            # Run any async initialization tasks here
            await self.validate_database_health()
            return True
        except Exception as e:
            logger.error(f"Async initialization error: {str(e)}")
            raise

    async def validate_database_health(self):
        """Validate database health during initialization."""
        from .utils.db_utils import validate_database, repair_database
        
        # Check database health
        is_valid, message = await validate_database(self.storage)
        if not is_valid:
            logger.warning(f"Database validation failed: {message}")
            
            # Attempt repair
            logger.info("Attempting database repair...")
            repair_success, repair_message = await repair_database(self.storage)
            
            if not repair_success:
                raise RuntimeError(f"Database repair failed: {repair_message}")
            else:
                logger.info(f"Database repair successful: {repair_message}")
        else:
            logger.info(f"Database validation successful: {message}")

async def async_main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    global CHROMA_PATH
    CHROMA_PATH = args.chroma_path
    
    logger.info(f"Starting MCP Memory Service with ChromaDB path: {CHROMA_PATH}")
    
    try:
        # Create server instance
        memory_server = MemoryServer()
        
        # Run async initialization
        await memory_server.initialize()
        
        # Start the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Server started and ready to handle requests")
            await memory_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    capabilities=memory_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise
import mongoose from 'mongoose';
import pino from 'pino';

const logger = pino({ transport: { target: 'pino-pretty' } });

export const connectDB = async () => {
  try {
    const uri = process.env.MONGO_URI || 'mongodb://localhost:27017/polymarket-bot';
    await mongoose.connect(uri);
    logger.info('MongoDB connected successfully');
  } catch (error) {
    logger.error({ err: error }, 'MongoDB connection failed');
    process.exit(1);
  }
};

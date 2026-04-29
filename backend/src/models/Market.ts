import mongoose, { Schema, Document } from 'mongoose';

export interface IMarket extends Document {
  conditionId: string;
  question: string;
  marketType: string;
  active: boolean;
  probabilities: { [key: string]: number };
  liquidity: number;
  volume24h: number;
  resolveDate?: Date;
  updatedAt: Date;
}

const MarketSchema = new Schema<IMarket>({
  conditionId:   { type: String, required: true, unique: true },
  question:      { type: String, default: '' },
  marketType:    { type: String, default: 'binary' },
  active:        { type: Boolean, default: true },
  probabilities: { type: Map, of: Number, default: {} },
  liquidity:     { type: Number, default: 0 },
  volume24h:     { type: Number, default: 0 },
  resolveDate:   { type: Date },
  updatedAt:     { type: Date, default: Date.now },
});

export default mongoose.model<IMarket>('Market', MarketSchema);

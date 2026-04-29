import mongoose, { Schema, Document } from 'mongoose';

export interface ITrade extends Document {
  tradeId: string;
  conditionId: string;
  assetId: string;
  position: 'YES' | 'NO';
  stakeAmount: number;
  entryPrice: number;
  status: 'OPEN' | 'CLOSED' | 'SETTLED';
  pnl: number;
  executedAt: Date;
  closedAt?: Date;
}

const TradeSchema = new Schema<ITrade>({
  tradeId:     { type: String, required: true, unique: true },
  conditionId: { type: String, required: true },
  assetId:     { type: String, default: 'UNKNOWN' },
  position:    { type: String, enum: ['YES', 'NO'], default: 'YES' },
  stakeAmount: { type: Number, default: 0 },
  entryPrice:  { type: Number, default: 0 },
  status:      { type: String, enum: ['OPEN', 'CLOSED', 'SETTLED'], default: 'OPEN' },
  pnl:         { type: Number, default: 0 },
  executedAt:  { type: Date, default: Date.now },
  closedAt:    { type: Date },
});

export default mongoose.model<ITrade>('Trade', TradeSchema);

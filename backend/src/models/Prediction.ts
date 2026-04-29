import mongoose, { Schema, Document } from 'mongoose';

export interface IPrediction extends Document {
  conditionId: string;
  modelType: string;
  trueProbability: number;
  confidence: number;
  marketProbabilityAtTime: number;
  expectedValue: number;
  externalSignals: any;
  timestamp: Date;
}

const PredictionSchema = new Schema<IPrediction>({
  conditionId:             { type: String, required: true },
  modelType:               { type: String, default: 'orchestrator' },
  trueProbability:         { type: Number, default: 0.5 },
  confidence:              { type: Number, default: 0 },
  marketProbabilityAtTime: { type: Number, default: 0.5 },
  expectedValue:           { type: Number, default: 0 },
  externalSignals:         { type: Schema.Types.Mixed, default: {} },
  timestamp:               { type: Date, default: Date.now },
});

export default mongoose.model<IPrediction>('Prediction', PredictionSchema);

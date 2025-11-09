-- Migration for narrative classification system
-- Story 1.4: Narrative Categorization and Token Validation

-- Add narrative classification fields to tokens table
ALTER TABLE tokens
ADD COLUMN IF NOT EXISTS classification_confidence DECIMAL(5,2) CHECK (classification_confidence >= 0 AND classification_confidence <= 100),
ADD COLUMN IF NOT EXISTS manual_review_status VARCHAR(20) DEFAULT 'pending' CHECK (manual_review_status IN ('pending', 'auto_classified', 'manual_reviewed', 'reviewed_approved', 'reviewed_rejected')),
ADD COLUMN IF NOT EXISTS reviewer VARCHAR(50),
ADD COLUMN IF NOT EXISTS review_date TIMESTAMP;

-- Update narrative_category constraint to include all valid categories
DO $$
BEGIN
    -- Drop existing constraint if it exists
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'tokens_narrative_category_check') THEN
        ALTER TABLE tokens DROP CONSTRAINT tokens_narrative_category_check;
    END IF;

    -- Add updated constraint
    ALTER TABLE tokens ADD CONSTRAINT tokens_narrative_category_check
    CHECK (narrative_category IN ('DeFi', 'Gaming', 'AI', 'Infrastructure', 'Meme', 'Stablecoin', 'RWA', 'Other'));
END $$;

-- Create indexes for narrative-based queries and analytics
CREATE INDEX IF NOT EXISTS idx_tokens_narrative_category ON tokens(narrative_category);
CREATE INDEX IF NOT EXISTS idx_tokens_manual_review_status ON tokens(manual_review_status);
CREATE INDEX IF NOT EXISTS idx_tokens_classification_confidence ON tokens(classification_confidence);
CREATE INDEX IF NOT EXISTS idx_tokens_reviewer ON tokens(reviewer);

-- Create audit table for manual classification decisions
CREATE TABLE IF NOT EXISTS classification_audit (
    id SERIAL PRIMARY KEY,
    token_address VARCHAR(42) NOT NULL,
    old_category VARCHAR(50),
    new_category VARCHAR(50) NOT NULL,
    old_confidence DECIMAL(5,2),
    new_confidence DECIMAL(5,2),
    reviewer VARCHAR(50) NOT NULL,
    review_reason TEXT,
    automated_suggestion VARCHAR(50),
    automated_confidence DECIMAL(5,2),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_audit_token FOREIGN KEY (token_address) REFERENCES tokens(token_address)
);

-- Create index for audit queries
CREATE INDEX IF NOT EXISTS idx_classification_audit_token ON classification_audit(token_address);
CREATE INDEX IF NOT EXISTS idx_classification_audit_reviewer ON classification_audit(reviewer);
CREATE INDEX IF NOT EXISTS idx_classification_audit_date ON classification_audit(created_at);

-- Create view for classification summary statistics
CREATE OR REPLACE VIEW classification_summary AS
SELECT
    COUNT(*) as total_tokens,
    COUNT(narrative_category) as classified_tokens,
    ROUND((COUNT(narrative_category)::DECIMAL / COUNT(*)) * 100, 2) as completeness_percentage,
    COUNT(CASE WHEN classification_confidence >= 80 THEN 1 END) as high_confidence_count,
    COUNT(CASE WHEN classification_confidence BETWEEN 50 AND 79.99 THEN 1 END) as medium_confidence_count,
    COUNT(CASE WHEN classification_confidence < 50 THEN 1 END) as low_confidence_count,
    COUNT(CASE WHEN manual_review_status = 'pending' THEN 1 END) as manual_review_pending,
    COUNT(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN 1 END) as manual_review_completed
FROM tokens;

-- Create view for category distribution
CREATE OR REPLACE VIEW category_distribution AS
SELECT
    narrative_category,
    COUNT(*) as token_count,
    ROUND((COUNT(*)::DECIMAL / (SELECT COUNT(*) FROM tokens WHERE narrative_category IS NOT NULL)) * 100, 2) as percentage,
    AVG(classification_confidence) as avg_confidence,
    COUNT(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN 1 END) as manually_reviewed_count
FROM tokens
WHERE narrative_category IS NOT NULL
GROUP BY narrative_category
ORDER BY token_count DESC;

-- Create view for manual review queue
CREATE OR REPLACE VIEW manual_review_queue AS
SELECT
    token_address,
    symbol,
    name,
    narrative_category,
    classification_confidence,
    manual_review_status,
    market_cap_rank,
    created_at,
    updated_at
FROM tokens
WHERE manual_review_status = 'pending'
   OR classification_confidence < 50
   OR narrative_category IS NULL
ORDER BY
    CASE WHEN market_cap_rank IS NOT NULL THEN market_cap_rank ELSE 999999 END ASC,
    classification_confidence ASC;

-- Create function to log classification changes
CREATE OR REPLACE FUNCTION log_classification_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Only log if narrative_category actually changed
    IF OLD.narrative_category IS DISTINCT FROM NEW.narrative_category
       OR OLD.classification_confidence IS DISTINCT FROM NEW.classification_confidence
       OR OLD.manual_review_status IS DISTINCT FROM NEW.manual_review_status THEN

        INSERT INTO classification_audit (
            token_address,
            old_category,
            new_category,
            old_confidence,
            new_confidence,
            reviewer,
            review_reason,
            automated_suggestion,
            automated_confidence
        ) VALUES (
            NEW.token_address,
            OLD.narrative_category,
            NEW.narrative_category,
            OLD.classification_confidence,
            NEW.classification_confidence,
            NEW.reviewer,
            CASE
                WHEN NEW.manual_review_status LIKE 'manual%' THEN 'Manual classification'
                WHEN NEW.manual_review_status = 'auto_classified' THEN 'Automated classification'
                WHEN NEW.manual_review_status LIKE 'reviewed%' THEN 'Manual review completed'
                ELSE 'Classification updated'
            END,
            CASE WHEN OLD.narrative_category IS NOT NULL THEN OLD.narrative_category ELSE NULL END,
            OLD.classification_confidence
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for classification audit logging
DROP TRIGGER IF EXISTS trigger_log_classification_change ON tokens;
CREATE TRIGGER trigger_log_classification_change
    AFTER UPDATE ON tokens
    FOR EACH ROW
    EXECUTE FUNCTION log_classification_change();

-- Add comments for documentation
COMMENT ON COLUMN tokens.narrative_category IS 'Narrative classification: DeFi, Gaming, AI, Infrastructure, Meme, Stablecoin, RWA, Other';
COMMENT ON COLUMN tokens.classification_confidence IS 'Confidence score (0-100) for narrative classification';
COMMENT ON COLUMN tokens.manual_review_status IS 'Status of manual review process for classification';
COMMENT ON COLUMN tokens.reviewer IS 'Username of person who reviewed/classified this token';
COMMENT ON COLUMN tokens.review_date IS 'Timestamp when manual review was completed';

COMMENT ON TABLE classification_audit IS 'Audit trail for all narrative classification changes';
COMMENT ON VIEW classification_summary IS 'Summary statistics for narrative classification completeness';
COMMENT ON VIEW category_distribution IS 'Distribution of tokens across narrative categories';
COMMENT ON VIEW manual_review_queue IS 'Tokens requiring manual review for classification';

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT ON classification_summary TO readonly_user;
-- GRANT SELECT ON category_distribution TO readonly_user;
-- GRANT SELECT ON manual_review_queue TO readonly_user;
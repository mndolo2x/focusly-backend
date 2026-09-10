-- Migration 002: Add image support to documents table

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS source_type VARCHAR(50) DEFAULT 'pdf',
ADD COLUMN IF NOT EXISTS original_images JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS virtual_page_map JSONB DEFAULT '[]';

-- Comment on columns
COMMENT ON COLUMN documents.source_type IS 'Type of document source: pdf or image_notes';
COMMENT ON COLUMN documents.original_images IS 'List of image storage URLs or file paths uploaded for note processing';
COMMENT ON COLUMN documents.virtual_page_map IS 'Mapping of virtual page numbers to original image metadata and extracted diagrams';

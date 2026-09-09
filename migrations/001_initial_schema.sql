-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Documents Table
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_url TEXT,
    extracted_text TEXT,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed', 'video_ready')),
    page_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Summaries Table
CREATE TABLE IF NOT EXISTS public.summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    depth TEXT NOT NULL DEFAULT 'quick' CHECK (depth IN ('quick', 'deep')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Quiz Questions Table
CREATE TABLE IF NOT EXISTS public.quiz_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    section_index INT NOT NULL DEFAULT 0,
    question_text TEXT NOT NULL,
    options JSONB NOT NULL DEFAULT '[]'::jsonb,
    correct_answer_index INT NOT NULL,
    page_reference INT
);

-- 4. Quiz Attempts Table
CREATE TABLE IF NOT EXISTS public.quiz_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES public.quiz_questions(id) ON DELETE CASCADE,
    selected_answer INT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    confidence_score INT CHECK (confidence_score BETWEEN 1 AND 5),
    answered_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Review Items Table (Spaced Repetition System)
CREATE TABLE IF NOT EXISTS public.review_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES public.quiz_questions(id) ON DELETE CASCADE,
    last_reviewed TIMESTAMPTZ DEFAULT NOW(),
    ease_factor FLOAT DEFAULT 2.5,
    interval INT DEFAULT 1,
    next_review_date TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Study Plans Table
CREATE TABLE IF NOT EXISTS public.study_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    exam_date TIMESTAMPTZ NOT NULL,
    subject TEXT,
    schedule JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Exam Profiles Table
CREATE TABLE IF NOT EXISTS public.exam_profiles (
    id TEXT PRIMARY KEY,
    exam_name TEXT NOT NULL,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    question_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    time_limit INT,
    total_questions INT DEFAULT 50,
    scoring_rules TEXT,
    disclaimer_text TEXT NOT NULL
);

-- 8. User Usage Table
CREATE TABLE IF NOT EXISTS public.user_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    month INT NOT NULL CHECK (month BETWEEN 1 AND 12),
    year INT NOT NULL,
    video_generations_used INT DEFAULT 0,
    summary_generations_used INT DEFAULT 0,
    UNIQUE(user_id, month, year)
);

-- 9. Shared Links Table
CREATE TABLE IF NOT EXISTS public.shared_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    share_id TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- --- INDEXES ---
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents(status);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_user_id ON public.quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_review_items_user_id ON public.review_items(user_id);
CREATE INDEX IF NOT EXISTS idx_review_items_next_review_date ON public.review_items(next_review_date);

-- --- ROW LEVEL SECURITY (RLS) ---
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.review_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.study_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.shared_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.exam_profiles ENABLE ROW LEVEL SECURITY;

-- Documents RLS Policies
CREATE POLICY "Users can select own documents" ON public.documents FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own documents" ON public.documents FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own documents" ON public.documents FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own documents" ON public.documents FOR DELETE USING (auth.uid() = user_id);

-- Summaries RLS Policies (via document ownership)
CREATE POLICY "Users can access summaries of own documents" ON public.summaries FOR ALL USING (
    EXISTS (SELECT 1 FROM public.documents WHERE documents.id = summaries.document_id AND documents.user_id = auth.uid())
);

-- Quiz Questions RLS Policies
CREATE POLICY "Users can access quiz questions of own documents" ON public.quiz_questions FOR ALL USING (
    EXISTS (SELECT 1 FROM public.documents WHERE documents.id = quiz_questions.document_id AND documents.user_id = auth.uid())
);

-- Quiz Attempts RLS Policies
CREATE POLICY "Users can select own quiz attempts" ON public.quiz_attempts FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own quiz attempts" ON public.quiz_attempts FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own quiz attempts" ON public.quiz_attempts FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own quiz attempts" ON public.quiz_attempts FOR DELETE USING (auth.uid() = user_id);

-- Review Items RLS Policies
CREATE POLICY "Users can select own review items" ON public.review_items FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own review items" ON public.review_items FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own review items" ON public.review_items FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own review items" ON public.review_items FOR DELETE USING (auth.uid() = user_id);

-- Study Plans RLS Policies
CREATE POLICY "Users can select own study plans" ON public.study_plans FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own study plans" ON public.study_plans FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own study plans" ON public.study_plans FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own study plans" ON public.study_plans FOR DELETE USING (auth.uid() = user_id);

-- User Usage RLS Policies
CREATE POLICY "Users can view own usage" ON public.user_usage FOR SELECT USING (auth.uid() = user_id);

-- Exam Profiles (Public Read)
CREATE POLICY "Anyone can view exam profiles" ON public.exam_profiles FOR SELECT USING (true);

-- Shared Links (Public Read for active shares)
CREATE POLICY "Anyone can view valid shared links" ON public.shared_links FOR SELECT USING (expires_at > NOW());
CREATE POLICY "Users can create shared links for own documents" ON public.shared_links FOR INSERT WITH CHECK (
    EXISTS (SELECT 1 FROM public.documents WHERE documents.id = shared_links.document_id AND documents.user_id = auth.uid())
);

-- --- FUNCTIONS & TRIGGERS ---

-- Function: get_user_usage
CREATE OR REPLACE FUNCTION public.get_user_usage(user_uuid UUID, month_val INT, year_val INT)
RETURNS TABLE (
    video_generations_used INT,
    summary_generations_used INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(u.video_generations_used, 0),
        COALESCE(u.summary_generations_used, 0)
    FROM public.user_usage u
    WHERE u.user_id = user_uuid AND u.month = month_val AND u.year = year_val;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger Function: handle_new_user_usage
CREATE OR REPLACE FUNCTION public.handle_new_user_usage()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_usage (user_id, month, year, video_generations_used, summary_generations_used)
    VALUES (
        NEW.id,
        EXTRACT(MONTH FROM CURRENT_DATE)::INT,
        EXTRACT(YEAR FROM CURRENT_DATE)::INT,
        0,
        0
    )
    ON CONFLICT (user_id, month, year) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger on auth.users creation
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user_usage();

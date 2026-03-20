"""
Template Learning Service
Learns from user corrections to improve document parsing accuracy
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json
import logging

logger = logging.getLogger(__name__)


class TemplateLearningService:
    """Service for learning from user corrections and improving document parsing"""
    
    async def save_correction(
        self,
        user_id: str,
        document_category: str,
        original_extraction: Dict[str, Any],
        corrected_data: Dict[str, Any],
        correction_type: str = "field_mapping",
        correction_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a user correction for learning
        
        Args:
            user_id: User who made the correction
            document_category: Category of document (invoice, receipt, etc.)
            original_extraction: Original extracted data
            corrected_data: User-corrected data
            correction_type: Type of correction (field_mapping, column_mapping, extraction_rule)
            correction_details: Additional details about the correction
        
        Returns:
            Correction ID
        """
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            correction_id = str(uuid.uuid4())
            
            # Find or create template
            template_id = await self._get_or_create_template(
                user_id, document_category, original_extraction
            )
            
            # Save correction
            await postgres.execute("""
                INSERT INTO cyrex.document_parsing_corrections
                (correction_id, user_id, template_id, document_category, original_extraction,
                 corrected_data, correction_type, correction_details, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            """,
                correction_id,
                user_id,
                template_id,
                document_category,
                json.dumps(original_extraction),
                json.dumps(corrected_data),
                correction_type,
                json.dumps(correction_details or {}),
            )
            
            # Update template statistics
            await self._update_template_stats(template_id, correction_type)
            
            logger.info(f"Saved correction {correction_id} for user {user_id}, category {document_category}")
            return correction_id
            
        except Exception as e:
            logger.error(f"Failed to save correction: {e}", exc_info=True)
            raise
    
    async def _get_or_create_template(
        self,
        user_id: str,
        document_category: str,
        extraction: Dict[str, Any],
    ) -> str:
        """Get or create a template for the document category"""
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            
            # Try to find existing template
            row = await postgres.fetchrow("""
                SELECT template_id FROM cyrex.document_parsing_templates
                WHERE user_id = $1 AND document_category = $2
                ORDER BY last_used_at DESC NULLS LAST, updated_at DESC
                LIMIT 1
            """, user_id, document_category)
            
            if row:
                template_id = row['template_id']
                # Update last_used_at
                await postgres.execute("""
                    UPDATE cyrex.document_parsing_templates
                    SET last_used_at = NOW(), updated_at = NOW()
                    WHERE template_id = $1
                """, template_id)
                return template_id
            
            # Create new template
            template_id = str(uuid.uuid4())
            await postgres.execute("""
                INSERT INTO cyrex.document_parsing_templates
                (template_id, user_id, document_category, template_name, field_mappings,
                 column_mappings, extraction_rules, created_at, updated_at, last_used_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW(), NOW())
            """,
                template_id,
                user_id,
                document_category,
                f"{document_category}_template",
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
            )
            
            return template_id
            
        except Exception as e:
            logger.error(f"Failed to get/create template: {e}", exc_info=True)
            raise
    
    async def _update_template_stats(self, template_id: str, correction_type: str):
        """Update template statistics after correction"""
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            
            await postgres.execute("""
                UPDATE cyrex.document_parsing_templates
                SET correction_count = correction_count + 1,
                    updated_at = NOW()
                WHERE template_id = $1
            """, template_id)
            
        except Exception as e:
            logger.warning(f"Failed to update template stats: {e}")
    
    async def learn_from_corrections(
        self,
        user_id: str,
        document_category: str,
    ) -> Dict[str, Any]:
        """
        Learn patterns from user corrections and update template
        
        Args:
            user_id: User ID
            document_category: Document category
        
        Returns:
            Updated template with learned patterns
        """
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            
            # Get recent corrections
            corrections = await postgres.fetch("""
                SELECT original_extraction, corrected_data, correction_type, correction_details
                FROM cyrex.document_parsing_corrections
                WHERE user_id = $1 AND document_category = $2
                ORDER BY created_at DESC
                LIMIT 50
            """, user_id, document_category)
            
            if not corrections:
                return {}
            
            # Analyze patterns
            field_mappings = {}
            column_mappings = {}
            extraction_rules = {}
            
            for correction in corrections:
                orig = json.loads(correction['original_extraction'])
                corrected = json.loads(correction['corrected_data'])
                corr_type = correction['correction_type']
                
                if corr_type == 'field_mapping':
                    # Learn field mappings
                    for key, value in corrected.items():
                        if key not in orig or orig[key] != value:
                            # Track common corrections
                            if key not in field_mappings:
                                field_mappings[key] = {'patterns': [], 'count': 0}
                            field_mappings[key]['count'] += 1
                            field_mappings[key]['patterns'].append({
                                'original': orig.get(key),
                                'corrected': value,
                            })
                
                elif corr_type == 'column_mapping':
                    # Learn column mappings
                    if 'columns' in corrected:
                        column_mappings.update(corrected['columns'])
            
            # Update template with learned patterns
            template_id = await self._get_or_create_template(user_id, document_category, {})
            
            postgres = await get_postgres_manager()
            await postgres.execute("""
                UPDATE cyrex.document_parsing_templates
                SET field_mappings = $1,
                    column_mappings = $2,
                    extraction_rules = $3,
                    updated_at = NOW()
                WHERE template_id = $4
            """,
                json.dumps(field_mappings),
                json.dumps(column_mappings),
                json.dumps(extraction_rules),
                template_id,
            )
            
            logger.info(f"Learned patterns from {len(corrections)} corrections for user {user_id}")
            
            return {
                'field_mappings': field_mappings,
                'column_mappings': column_mappings,
                'extraction_rules': extraction_rules,
            }
            
        except Exception as e:
            logger.error(f"Failed to learn from corrections: {e}", exc_info=True)
            return {}
    
    async def get_template(
        self,
        user_id: str,
        document_category: str,
    ) -> Optional[Dict[str, Any]]:
        """Get learned template for document category"""
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            
            row = await postgres.fetchrow("""
                SELECT template_id, field_mappings, column_mappings, extraction_rules,
                       correction_count, success_count, last_used_at
                FROM cyrex.document_parsing_templates
                WHERE user_id = $1 AND document_category = $2
                ORDER BY last_used_at DESC NULLS LAST, updated_at DESC
                LIMIT 1
            """, user_id, document_category)
            
            if not row:
                return None
            
            return {
                'template_id': row['template_id'],
                'field_mappings': json.loads(row['field_mappings']) if row['field_mappings'] else {},
                'column_mappings': json.loads(row['column_mappings']) if row['column_mappings'] else {},
                'extraction_rules': json.loads(row['extraction_rules']) if row['extraction_rules'] else {},
                'correction_count': row['correction_count'],
                'success_count': row['success_count'],
                'last_used_at': row['last_used_at'].isoformat() if row['last_used_at'] else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get template: {e}", exc_info=True)
            return None
    
    async def mark_success(self, template_id: str):
        """Mark a successful extraction using this template"""
        try:
            from ..database.postgres import get_postgres_manager
            
            postgres = await get_postgres_manager()
            
            await postgres.execute("""
                UPDATE cyrex.document_parsing_templates
                SET success_count = success_count + 1,
                    last_used_at = NOW(),
                    updated_at = NOW()
                WHERE template_id = $1
            """, template_id)
            
        except Exception as e:
            logger.warning(f"Failed to mark success: {e}")


# Singleton instance
_learning_service: Optional[TemplateLearningService] = None


def get_template_learning_service() -> TemplateLearningService:
    """Get singleton TemplateLearningService instance"""
    global _learning_service
    if _learning_service is None:
        _learning_service = TemplateLearningService()
    return _learning_service


import torch
import torch.nn.functional as F

from kd_pipeline.distill import kd_loss, train_knowledge_distillation
from kd_pipeline.models import LightNN


def test_kd_loss_matches_manual_computation():
    torch.manual_seed(0)
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    T, soft_w, ce_w = 2.0, 0.25, 0.75

    loss = kd_loss(
        student_logits, teacher_logits, labels, T=T, soft_target_loss_weight=soft_w, ce_loss_weight=ce_w
    )

    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    soft_prob = F.log_softmax(student_logits / T, dim=-1)
    expected_kd = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (T**2)
    expected_ce = F.cross_entropy(student_logits, labels)
    expected = soft_w * expected_kd + ce_w * expected_ce

    assert torch.allclose(loss, expected, atol=1e-6)


def test_kd_loss_is_positive_scalar():
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = kd_loss(
        student_logits, teacher_logits, labels, T=2.0, soft_target_loss_weight=0.25, ce_loss_weight=0.75
    )
    assert loss.dim() == 0
    assert loss.item() > 0


def test_teacher_receives_no_gradient_during_distillation():
    from torch.utils.data import DataLoader, TensorDataset

    teacher = LightNN(num_classes=10)
    student = LightNN(num_classes=10)

    n = 8
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=loader,
        val_loader=loader,
        epochs=1,
        learning_rate=0.001,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device="cpu",
    )

    for p in teacher.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
